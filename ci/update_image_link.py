import requests
import sys


def update_link(link_id, new_link, api_key):
    response = requests.patch(
        f'https://api.dub.co/links/{link_id}',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        json={
            'url': new_link
        }
    )

    if response.status_code == 200:
        print("Link updated successfully.")
    else:
        raise Exception(f"Error updating link: {response.text}")


if __name__ == '__main__':
    link_id = sys.argv[1]
    new_link = sys.argv[2]
    api_key = sys.argv[3]
    update_link(link_id, new_link, api_key)


