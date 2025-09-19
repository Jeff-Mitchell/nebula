import argparse
import json
import sys

import requests


# load scenario from JSON file
def load_scenario(scenario_path):
    try:
        with open(scenario_path) as f:
            return json.load(f)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


# get authentication token
def get_session_cookie(port):
    try:
        session = requests.Session()
        response = session.post(
            f"http://localhost:{port}/platform/login",
            data={"user": "admin", "password": "admin"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code == 200:
            cookie = session.cookies.get("session_None")
            if cookie:
                return cookie

        else:
            return None

    except Exception:
        return None


# submit scenario to frontend
def submit_scenario(scenario_data, port):
    url = f"http://localhost:{port}/platform/dashboard/deployment/run"

    try:
        response = requests.post(
            url,
            json=[scenario_data],
            headers={"Content-Type": "application/json"},
            cookies={"session_None": get_session_cookie(port)},
            timeout=30,
            allow_redirects=False,
        )

        if response.status_code == 303:
            print("Success: Scenario submitted")
            return True

        elif response.status_code == 401:
            print("Error: Authentication failed")
            return False

        else:
            print(f"Error: HTTP {response.status_code} - {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to frontend at localhost:{port}")
        return False

    except Exception as e:
        print(f"Error: {e}")
        return False


# main entrypoint
def main():
    parser = argparse.ArgumentParser(description="Submit NEBULA scenario via frontend API")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON file")
    parser.add_argument("--port", type=int, default=6060, help="Frontend port (default: 6060)")
    args = parser.parse_args()

    scenario_data = load_scenario(args.scenario)

    if submit_scenario(scenario_data, args.port):
        print("Success: Scenario accepted by frontend")

    else:
        print("Error: Failed to submit scenario")
        sys.exit(1)


if __name__ == "__main__":
    main()
