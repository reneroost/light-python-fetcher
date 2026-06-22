import sys
from curl_cffi import requests


def main() -> None:
    # 1. Accept target domain input from the user
    try:
        domain_input = (
            input("Enter the domain name to test (e.g., example): ")
            .strip()
            .lower()
        )
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled by user.")
        sys.exit(0)

    if not domain_input:
        print("Error: Domain name cannot be empty.")
        sys.exit(1)

    # 2. Construct the target URL
    target_url = f"https://{domain_input}.com"

    # 3. Clearly log the site being requested
    print(f"\n[+] Requesting: {target_url} using chrome impersonation...")

    try:
        # 4. Execute the impersonated request
        response = requests.get(target_url, impersonate="chrome")

        # 5. Output results
        print(f"[+] Status Code: {response.status_code}")
        print("[+] Response Preview (First 200 chars):")
        print("-" * 40)
        print(response.text[:200])
        print("-" * 40)

    except Exception as e:
        print(f"[-] Request failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
