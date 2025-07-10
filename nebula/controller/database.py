import logging
import os
import psycopg2
import psycopg2.extras
from passlib.context import CryptContext
import datetime
import json
import asyncpg
import asyncio

from nebula.controller.scenarios import Scenario

# --- Configuration ---
# Use environment variables for database credentials from the Docker Compose file
DATABASE_URL = f"postgresql://{os.environ.get('DB_USER')}:{os.environ.get('DB_PASSWORD')}@{os.environ.get('DB_HOST')}:{os.environ.get('DB_PORT')}/nebula"

# Password hashing context (using Argon2)
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# Asynchronous lock for node updates
_node_lock = asyncio.Lock()


# --- Connection Management Helper Functions ---

async def get_async_conn():
    """Establishes an asynchronous PostgreSQL connection."""
    return await asyncpg.connect(DATABASE_URL)


# --- User Management Functions ---

async def list_users(all_info=False):
    """
    Retrieves a list of users from the users database.
    """
    conn = await get_async_conn()
    try:
        result = await conn.fetch("SELECT * FROM users")
    finally:
        await conn.close()

    if not all_info:
        result = [user["user"] for user in result]

    return result


async def get_user_info(user):
    """
    Fetches detailed information for a specific user from the users database.
    """
    conn = await get_async_conn()
    try:
        result = await conn.fetchrow('SELECT * FROM users WHERE "user" = $1', user)
    finally:
        await conn.close()
    return result


async def verify(user, password):
    """
    Verifies whether the provided password matches the stored hashed password for a user.
    """
    conn = await get_async_conn()
    try:
        result = await conn.fetchrow('SELECT password FROM users WHERE "user" = $1', user)
        if result:
            try:
                return pwd_context.verify(password, result[0])
            except Exception:
                # Catch more general exceptions during verification to be safe
                logging.error(f"Error during password verification for user {user}", exc_info=True)
                return False
    finally:
        await conn.close()
    return False


async def verify_hash_algorithm(user):
    """
    Checks if the stored password hash for a user uses a supported Argon2 algorithm.
    """
    user = user.upper()
    argon2_prefixes = ("$argon2i$", "$argon2id$")
    conn = await get_async_conn()
    try:
        result = await conn.fetchrow('SELECT password FROM users WHERE "user" = $1', user)
        if result:
            password_hash = result["password"]
            return password_hash.startswith(argon2_prefixes)
    finally:
        await conn.close()
    return False


async def delete_user_from_db(user):
    """
    Deletes a user record from the users database.
    """
    conn = await get_async_conn()
    try:
        await conn.execute('DELETE FROM users WHERE "user" = $1', user)
    finally:
        await conn.close()


async def add_user(user, password, role):
    """
    Adds a new user to the users database with a hashed password.
    """
    conn = await get_async_conn()
    try:
        hashed_password = pwd_context.hash(password)
        await conn.execute(
            'INSERT INTO users ("user", password, role) VALUES ($1, $2, $3)',
            user.upper(), hashed_password, role,
        )
    finally:
        await conn.close()


async def update_user(user, password, role):
    """
    Updates the password and role of an existing user in the users database.
    """
    conn = await get_async_conn()
    try:
        hashed_password = pwd_context.hash(password)
        await conn.execute(
            'UPDATE users SET password = $1, role = $2 WHERE "user" = $3',
            hashed_password, role, user.upper(),
        )
    finally:
        await conn.close()

# --- Node Management Functions ---

async def list_nodes(scenario_name=None, sort_by="idx"):
    """
    Retrieves a list of nodes from the nodes database, optionally filtered by scenario and sorted.
    """
    conn = await get_async_conn()
    try:
        # Validate sort_by to prevent SQL injection
        allowed_sort_fields = ["uid", "idx", "ip", "port", "role", "timestamp", "federation", "round"]
        if sort_by not in allowed_sort_fields:
            sort_by = "idx" # Default to a safe field

        if scenario_name:
            # Using f-string for column names is generally safe if validated as above
            command = f"SELECT * FROM nodes WHERE scenario = $1 ORDER BY {sort_by};"
            result = await conn.fetch(command, scenario_name)
        else:
            command = f"SELECT * FROM nodes ORDER BY {sort_by};"
            result = await conn.fetch(command)

        return result
    except asyncpg.PostgresError as e:
        logging.error(f"Error occurred while listing nodes: {e}")
        return None
    finally:
        await conn.close()


async def list_nodes_by_scenario_name(scenario_name):
    """
    Fetches all nodes associated with a specific scenario, ordered by their index as integers.
    """
    conn = None
    try:
        conn = await get_async_conn()
        command = "SELECT * FROM nodes WHERE scenario = $1 ORDER BY CAST(idx AS INTEGER) ASC;"
        result = await conn.fetch(command, scenario_name)
        return [dict(record) for record in result]
    except Exception as e:
        logging.error(f"Error occurred while listing nodes by scenario name: {e}")
        return None
    finally:
        if conn:
            await conn.close()


async def update_node_record(
    node_uid, idx, ip, port, role, neighbors, latitude, longitude,
    timestamp, federation, federation_round, scenario, run_hash, malicious,
):
    """
    Inserts or updates a node record in the database for a given scenario, ensuring thread-safe access.
    """
    async with _node_lock:
        # Await the get_async_conn() call to get the actual connection object
        conn = await get_async_conn()
        try:
            async with conn.transaction():
                result = await conn.fetchrow(
                    "SELECT * FROM nodes WHERE uid = $1 AND scenario = $2 FOR UPDATE;",
                    node_uid, scenario
                )

                if result is None:
                    # Insert new node
                    await conn.execute(
                        """
                        INSERT INTO nodes (uid, idx, ip, port, role, neighbors, latitude, longitude,
                                           timestamp, federation, round, scenario, hash, malicious)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14);
                        """,
                        node_uid, idx, ip, port, role, neighbors, latitude, longitude,
                        timestamp, federation, federation_round, scenario, run_hash, malicious,
                    )
                else:
                    # Update existing node
                    await conn.execute(
                        """
                        UPDATE nodes SET idx = $1, ip = $2, port = $3, role = $4, neighbors = $5,
                        latitude = $6, longitude = $7, timestamp = $8, federation = $9, round = $10,
                        hash = $11, malicious = $12
                        WHERE uid = $13 AND scenario = $14;
                        """,
                        idx, ip, port, role, neighbors, latitude, longitude,
                        timestamp, federation, federation_round, run_hash, malicious,
                        node_uid, scenario,
                    )

                updated_row = await conn.fetchrow("SELECT * from nodes WHERE uid = $1 AND scenario = $2;", node_uid, scenario)
                return dict(updated_row) if updated_row else None
        finally:
            # Ensure the connection is closed after use
            await conn.close()


async def remove_all_nodes():
    """
    Deletes all node records from the nodes database.
    """
    conn = await get_async_conn()
    try:
        await conn.execute("TRUNCATE nodes CASCADE;") # Use CASCADE if there are foreign key dependencies
    finally:
        await conn.close()


async def remove_nodes_by_scenario_name(scenario_name):
    """
    Deletes all nodes associated with a specific scenario from the database.
    """
    conn = await get_async_conn()
    try:
        await conn.execute("DELETE FROM nodes WHERE scenario = $1;", scenario_name)
    finally:
        await conn.close()

# --- Scenario Management Functions ---

async def get_all_scenarios(username, role, sort_by="start_time"):
    """
    Retrieves all scenarios from the database, accessing fields from the 'config' (JSONB) column
    and direct columns. Filters by user role and sorts by the specified field.
    """
    allowed_sort_fields = ["start_time", "title", "username", "status", "name"]
    if sort_by not in allowed_sort_fields:
        sort_by = "start_time"

    # Determine the ORDER BY clause based on sort_by
    if sort_by == "start_time":
        order_by_clause = """
            ORDER BY
                CASE
                    WHEN start_time IS NULL OR start_time = '' THEN 1
                    ELSE 0
                END,
                to_timestamp(start_time, 'YYYY/MM/DD HH24:MI:SS') DESC
        """
    elif sort_by in ["title", "model", "dataset", "rounds"]: # These are inside config JSONB
        order_by_clause = f"ORDER BY config->>'{sort_by}'"
    else: # For direct table columns like name, username, status
        order_by_clause = f"ORDER BY {sort_by}"

    conn = await get_async_conn()
    try:
        # Select direct columns and relevant fields from config JSONB
        command = """
            SELECT
                name,
                username,
                status,
                start_time,
                end_time,
                config->>'title' AS title,
                config->>'model' AS model,
                config->>'dataset' AS dataset,
                config->>'rounds' AS rounds,
                config -- return the full config JSONB
            FROM scenarios
        """
        params = []

        if role != "admin":
            command += " WHERE username = $1" # username is a direct column now
            params.append(username)

        full_command = f"{command} {order_by_clause};"
        result = await conn.fetch(full_command, *params)
    finally:
        await conn.close()

    return result


async def get_all_scenarios_and_check_completed(username, role, sort_by="start_time"):
    """
    Retrieves all scenarios, sorts them, and updates the status if necessary.
    Returns a list of dictionaries, where each dictionary represents a scenario.
    """
    # Safe list of allowed sorting fields to prevent SQL injection.
    allowed_sort_fields = ["start_time", "title", "username", "status", "name"]
    if sort_by not in allowed_sort_fields:
        sort_by = "start_time"  # Safe default value

    # Building the ORDER BY clause (same as get_all_scenarios)
    if sort_by == "start_time":
        order_by_clause = """
            ORDER BY
                CASE
                    WHEN start_time IS NULL OR start_time = '' THEN 1
                    ELSE 0
                END,
                -- CORRECTED: Changed 'DD/MM/YYYY' to 'YYYY/MM/DD' to match the storage format
                to_timestamp(start_time, 'DD/MM/YYYY HH24:MI:SS') DESC
        """
    elif sort_by in ["title", "model", "dataset", "rounds"]: # These are inside config JSONB
        order_by_clause = f"ORDER BY config->>'{sort_by}'"
    else: # For direct table columns like name, username, status
        order_by_clause = f"ORDER BY {sort_by}"

    conn = await get_async_conn()
    try:
        # Base query that extracts fields from the JSONB using the ->> operator
        command = f"""
            SELECT
                name,
                username,
                status,
                start_time,
                end_time,
                config->>'title' AS title,
                config->>'model' AS model,
                config->>'dataset' AS dataset,
                config->>'rounds' AS rounds,
                config  -- Return the full config object
            FROM scenarios
        """
        params = []
        if role != "admin":
            command += " WHERE username = $1" # username is a direct column
            params.append(username)

        command += f" {order_by_clause};"

        result_dicts = await conn.fetch(command, *params)

        scenarios_to_return = [dict(s) for s in result_dicts]

        re_fetch_required = False
        for scenario in scenarios_to_return:
            if scenario["status"] == "running":
                if await check_scenario_federation_completed(scenario["name"]):
                    await scenario_set_status_to_completed(scenario["name"])
                    re_fetch_required = True
                    break

        if re_fetch_required:
            # Recursively call get_all_scenarios_and_check_completed to get fresh data
            return await get_all_scenarios_and_check_completed(username, role, sort_by)
    finally:
        await conn.close()

    return scenarios_to_return


async def scenario_update_record(name, start_time, end_time, scenario_config, status, username):
    """
    Inserts or updates a scenario record using the PostgreSQL "UPSERT" pattern.
    All configuration is saved in the 'config' column of type JSONB.
    Direct columns (name, start_time, end_time, username, status) are also handled.
    """
    conn = await get_async_conn()
    try:
        # Ensure scenario_config is a dictionary before dumping to JSON
        if not isinstance(scenario_config, dict):
            try:
                scenario_config = json.loads(scenario_config)
            except json.JSONDecodeError:
                logging.error("scenario_config is not a valid JSON string or dict.")
                return

        command = """
            INSERT INTO scenarios (name, start_time, end_time, username, status, config)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            ON CONFLICT (name) DO UPDATE SET
                start_time = EXCLUDED.start_time,
                end_time = EXCLUDED.end_time,
                username = EXCLUDED.username,
                status = EXCLUDED.status,
                config = scenarios.config || EXCLUDED.config; -- Merge JSONB
        """
        await conn.execute(command, name, start_time, end_time, username, status, json.dumps(scenario_config))
    finally:
        await conn.close()


async def scenario_set_all_status_to_finished():
    """
    Sets the status of all 'running' scenarios to 'finished'
    and updates their 'end_time' (both in the direct column and within JSONB).
    """
    conn = await get_async_conn()
    try:
        current_time = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S') # Consistent format
        # Update direct columns first, then update JSONB within config
        command = """
            UPDATE scenarios
            SET
                status = 'finished',
                end_time = $1,
                config = jsonb_set(config, '{status}', '"finished"') ||
                         jsonb_set(config, '{end_time}', $2::jsonb)
            WHERE status = 'running';
        """
        await conn.execute(command, current_time, json.dumps(current_time))
    finally:
        await conn.close()


async def scenario_set_status_to_finished(scenario_name):
    """
    Sets the status of a specific scenario to 'finished' and updates its 'end_time'.
    Updates both the direct columns and the JSONB 'config'.
    """
    conn = await get_async_conn()
    try:
        current_time = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S') # Consistent format
        command = """
            UPDATE scenarios
            SET
                status = 'finished',
                end_time = $1,
                config = jsonb_set(
                             jsonb_set(config, '{status}', '"finished"'),
                             '{end_time}', $2::jsonb
                           )
            WHERE name = $3;
        """
        await conn.execute(command, current_time, json.dumps(current_time), scenario_name)
    finally:
        await conn.close()


async def scenario_set_status_to_completed(scenario_name):
    """
    Sets the status of a specific scenario to 'completed'.
    Updates both the direct column and the JSONB 'config'.
    """
    conn = await get_async_conn()
    try:
        command = """
            UPDATE scenarios
            SET
                status = 'completed',
                config = jsonb_set(config, '{status}', '"completed"')
            WHERE name = $1;
        """
        await conn.execute(command, scenario_name)
    finally:
        await conn.close()


async def get_running_scenario(username=None, get_all=False):
    """
    Retrieves scenarios with a 'running' status, optionally filtered by user.
    Returns full scenario record (including direct columns and config JSONB).
    """
    conn = await get_async_conn()
    try:
        params = ["running"]
        # Select all columns to get both direct and config data
        command = "SELECT name, username, status, start_time, end_time, config FROM scenarios WHERE status = $1"

        if username:
            command += " AND username = $2"
            params.append(username)

        if get_all:
            result = [dict(row) for row in await conn.fetch(command, *params)] # Convert records to dicts
        else:
            result_row = await conn.fetchrow(command, *params)
            result = dict(result_row) if result_row else None
    finally:
        await conn.close()
    return result


async def get_completed_scenario():
    """
    Retrieves a single scenario with a 'completed' status.
    Returns full scenario record (including direct columns and config JSONB).
    """
    conn = await get_async_conn()
    try:
        # The status is now a direct column, not just in config->>'status'
        command = "SELECT name, username, status, start_time, end_time, config FROM scenarios WHERE status = $1;"
        result_row = await conn.fetchrow(command, "completed")
        result = dict(result_row) if result_row else None
    finally:
        await conn.close()
    return result


async def get_scenario_by_name(scenario_name):
    """
    Retrieves the complete record of a scenario by its name.
    """
    conn = await get_async_conn()
    try:
        result_row = await conn.fetchrow("SELECT name, start_time, end_time, username, status, config FROM scenarios WHERE name = $1;", scenario_name)
        result = dict(result_row) if result_row else None

        if result and result.get('config'):
            # Assuming 'config' is a JSON string, so we parse it
            config_data = json.loads(result['config'])

            # Extract the 'scenario_title' and add it as a top-level key
            # Use .get() for safety in case 'scenario_title' is also missing within config
            result['title'] = config_data.get('scenario_title')

            # Also, if 'description' is inside config, you'll need to extract it similarly
            result['description'] = config_data.get('description') # Assuming 'description' is also in config
    finally:
        await conn.close()
    return result


async def get_user_by_scenario_name(scenario_name):
    """
    Retrieves the username associated with a scenario (from the direct 'username' column).
    """
    conn = await get_async_conn()
    try:
        result = await conn.fetchval("SELECT username FROM scenarios WHERE name = $1;", scenario_name)
    finally:
        await conn.close()
    return result


async def remove_scenario_by_name(scenario_name):
    """
    Delete a scenario from the database by its unique name.

    Parameters:
        scenario_name (str): The unique name identifier of the scenario to be removed.

    Behavior:
        - Removes the scenario record matching the given name.
        - Commits the deletion to the database.
    """
    conn = await get_async_conn()
    try:
        await conn.execute("DELETE FROM scenarios WHERE name = $1;", scenario_name)
        logging.info(f"Scenario '{scenario_name}' successfully removed.")
    except asyncpg.PostgresError as e:
        logging.error(f"Error occurred while deleting scenario '{scenario_name}': {e}")
    finally:
        await conn.close()


async def check_scenario_federation_completed(scenario_name):
    """
    Check if all nodes in a given scenario have completed the required federation rounds.

    Parameters:
        scenario_name (str): The unique name identifier of the scenario to check.

    Returns:
        bool: True if all nodes have completed the total rounds specified for the scenario, False otherwise or if an error occurs.

    Behavior:
        - Retrieves the total number of rounds defined for the scenario.
        - Fetches the current round progress of all nodes in that scenario.
        - Returns True only if every node has reached the total rounds.
        - Handles database errors and missing scenario cases gracefully.
    """
    conn = await get_async_conn()
    try:
        # Retrieve the total rounds for the scenario from the 'config' JSONB column
        scenario_rounds_str = await conn.fetchval("SELECT config->>'rounds' AS rounds FROM scenarios WHERE name = $1;", scenario_name)

        if not scenario_rounds_str:
            logging.warning(f"Scenario '{scenario_name}' not found or 'rounds' not defined.")
            return False

        # Ensure total_rounds is an integer for comparison
        try:
            total_rounds = int(scenario_rounds_str)
        except (ValueError, TypeError):
            logging.error(f"Invalid 'rounds' value for scenario '{scenario_name}': {scenario_rounds_str}")
            return False

        # Fetch the current round progress of all nodes in that scenario
        # The 'round' column in 'nodes' is a direct column
        nodes = await conn.fetch("SELECT round FROM nodes WHERE scenario = $1;", scenario_name)

        if not nodes:
            logging.info(f"No nodes found for scenario '{scenario_name}'. Federation not considered completed.")
            return False

        # Check if all nodes have completed the total rounds
        # The 'round' column in nodes is likely stored as a string or a numeric type.
        # Assuming 'round' in 'nodes' is a numeric type, we convert it to int for comparison.
        return all(int(node["round"]) >= total_rounds for node in nodes)

    except asyncpg.PostgresError as e:
        logging.error(f"PostgreSQL error during check_scenario_federation_completed for scenario '{scenario_name}': {e}")
        return False
    except ValueError as e:
        logging.error(f"Data error during check_scenario_federation_completed for scenario '{scenario_name}': {e}")
        return False
    finally:
        await conn.close()


async def check_scenario_with_role(role, scenario_name, current_username=None):
    """
    Verify if a scenario exists that the user with the given role and username can access.

    Parameters:
        role (str): The role of the current user (e.g., "admin", "user").
        scenario_name (str): The unique name identifier of the scenario to check.
        current_username (str, optional): The username of the currently authenticated user.
                                          Required for non-admin roles.

    Returns:
        bool: True if the scenario exists and the user has access, False otherwise.

    Behavior:
        - If the user's role is "admin", they can access any existing scenario.
        - If the user's role is not "admin", they can only access scenarios where the
          scenario's 'username' matches their `current_username`.
    """
    scenario_info = await get_scenario_by_name(scenario_name)

    if not scenario_info:
        return False  # Scenario does not exist

    if role == "admin":
        return True  # Admins can access any existing scenario
    else:
        # For non-admin roles, check if the scenario's username matches the current user's username
        if current_username is None:
            logging.warning(
                "`check_scenario_with_role` called for non-admin role without `current_username`. "
                "Cannot verify user-specific scenario access."
            )
            return False # Cannot verify access without the current user's username

        return scenario_info.get("username") == current_username

# --- Notes Management Functions ---

async def save_notes(scenario, notes):
    """
    Save or update notes associated with a specific scenario.
    """
    conn = await get_async_conn()
    try:
        await conn.execute(
            """
            INSERT INTO notes (scenario, scenario_notes) VALUES ($1, $2)
            ON CONFLICT(scenario) DO UPDATE SET scenario_notes = EXCLUDED.scenario_notes;
            """,
            scenario, notes,
        )
    except asyncpg.IntegrityConstraintViolationError as e:
        logging.error(f"PostgreSQL integrity error during save_notes: {e}")
    except asyncpg.PostgresError as e:
        logging.error(f"PostgreSQL error during save_notes: {e}")
    finally:
        await conn.close()


async def get_notes(scenario):
    """
    Retrieve notes associated with a specific scenario.
    """
    conn = await get_async_conn()
    try:
        result = await conn.fetchrow("SELECT * FROM notes WHERE scenario = $1;", scenario)
    finally:
        await conn.close()
    return result


async def remove_note(scenario):
    """
    Delete the note associated with a specific scenario.
    """
    conn = await get_async_conn()
    try:
        await conn.execute("DELETE FROM notes WHERE scenario = $1;", scenario)
    finally:
        await conn.close()


if __name__ == "__main__":
    """
    Entry point for the script to print the list of users.
    """
    # Example usage (assuming DB_USER, DB_PASSWORD, DB_HOST, DB_PORT are set in env)
    # os.environ['DB_USER'] = 'your_db_user'
    # os.environ['DB_PASSWORD'] = 'your_db_password'
    # os.environ['DB_HOST'] = 'localhost'
    # os.environ['DB_PORT'] = '5432'

    logging.basicConfig(level=logging.INFO)
