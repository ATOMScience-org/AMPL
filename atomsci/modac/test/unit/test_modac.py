import json
import unittest
from unittest.mock import patch, MagicMock

import requests

from atomsci.modac import MoDaCClient, ensure_authenticated, MoDaCClientError


class TestMoDaCClient(unittest.TestCase):

    @patch("atomsci.modac.MoDaCClient._login_headers", return_value={})
    @patch("atomsci.modac.requests.get")
    def test_authenticate(self, mock_get, mock_login_headers):
        mock_resp = MagicMock()
        mock_resp.content.decode.return_value = "mock_token"
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        client = MoDaCClient()
        self.assertEqual(client._token, "mock_token")
        mock_get.assert_called_once_with(f"{client.BASE_URL}/authenticate", auth={})

    @patch.object(MoDaCClient, "authenticate", return_value=True)
    def test_ensure_authenticated_decorator(self, mock_authenticate):
        client = MoDaCClient()
        client._token = "mock_token"

        @ensure_authenticated
        def sample_method(self):
            return True

        sample_method(client)
        mock_authenticate.assert_called_once()

    @patch.object(MoDaCClient, "authenticate", return_value=True)
    def test_ensure_authenticated_decorator_no_token(self, mock_authenticate):
        client = MoDaCClient()
        client._token = ""

        @ensure_authenticated
        def sample_method(self):
            return True

        sample_method(client)
        self.assertEqual(mock_authenticate.call_count, 2)

    @patch("atomsci.modac.requests.post")
    @patch("atomsci.modac.MoDaCClient._token_headers", return_value={})
    @patch.object(MoDaCClient, "authenticate", return_value=True)
    def test_download_file(self, mock_authenticate, mock_token_headers, mock_post):
        client = MoDaCClient()
        client._token = "mock_token"

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
            client.download_file("mock_file_path", "mock_local_filename")
            mock_post.assert_called_once_with(
                f"{client.BASE_URL}/v2/dataObject/mock_file_path/download",
                headers={},
                json={},
            )
            mock_file.assert_called_once_with("mock_local_filename", "wb")
            mock_file().write.assert_called_once_with(mock_resp.content)

    @patch("atomsci.modac.MoDaCClient.get_collection")
    @patch("atomsci.modac.MoDaCClient.download_file")
    @patch("os.makedirs")
    @patch.object(MoDaCClient, "authenticate", return_value=True)
    def test_download_all_files_in_collection(
        self, mock_authenticate, mock_makedirs, mock_download_file, mock_get_collection
    ):
        client = MoDaCClient()
        client._token = "mock_token"

        mock_get_collection.return_value = {
            "collectionName": "mock_collection",
            "dataObjects": [{"path": "mock_file_path"}],
            "subCollections": [],
        }

        client.download_all_files_in_collection("mock_path")
        mock_makedirs.assert_called_once_with("mock_collection")
        mock_download_file.assert_called_once_with(
            "mock_file_path", "mock_collection/mock_file_path"
        )

    @patch("atomsci.modac.requests.get")
    @patch("atomsci.modac.MoDaCClient._login_headers", return_value={})
    def test_authenticate_server_error(self, mock_login_headers, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=MagicMock(status_code=502)
        )
        mock_resp.content.decode.return_value = "Bad Gateway"
        mock_get.return_value = mock_resp

        with self.assertRaises(MoDaCClientError) as context:
            client = MoDaCClient()
            mock_get.assert_called_once_with(f"{client.BASE_URL}/authenticate", auth={})

        self.assertIn("Authentication failed: 502", str(context.exception))

    @patch("atomsci.modac.requests.put")
    @patch("atomsci.modac.MoDaCClient._login_headers", return_value={})
    @patch(
        "builtins.open", new_callable=unittest.mock.mock_open, read_data=b"file content"
    )
    @patch.object(MoDaCClient, "authenticate", return_value=True)
    def test_upload_file_success(
        self, mock_authenticate, mock_open, mock_token_headers, mock_put
    ):
        client = MoDaCClient()

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"status": "success"}
        mock_put.return_value = mock_resp

        result = client.upload_file("mock_collection", "mock_data_file.txt")

        self.assertTrue(result)
        mock_put.assert_called_once()
        mock_open.assert_called_once_with("mock_data_file.txt", "rb")

    @patch("atomsci.modac.requests.put")
    @patch("atomsci.modac.MoDaCClient._login_headers", return_value={})
    @patch(
        "builtins.open", new_callable=unittest.mock.mock_open, read_data=b"file content"
    )
    @patch.object(MoDaCClient, "authenticate", return_value=True)
    def test_upload_file_with_attributes(
        self, mock_authenticate, mock_open, mock_token_headers, mock_put
    ):
        client = MoDaCClient()

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"status": "success"}
        mock_put.return_value = mock_resp

        attributes = {"attribute1": "value1", "attribute2": "value2"}

        result = client.upload_file(
            "mock_collection", "mock_data_file.txt", attributes_file=attributes
        )

        self.assertTrue(result)
        mock_put.assert_called_once()
        mock_open.assert_called_once_with("mock_data_file.txt", "rb")

        called_files = mock_put.call_args[1]["files"]
        self.assertIn("dataObjectRegistration", called_files)
        self.assertEqual(
            json.loads(called_files["dataObjectRegistration"][1]), attributes
        )

    @patch("atomsci.modac.requests.put")
    @patch(
        "builtins.open", new_callable=unittest.mock.mock_open, read_data=b"file content"
    )
    @patch.object(MoDaCClient, "authenticate", return_value=True)
    def test_upload_file_failure(self, mock_authenticate, mock_open, mock_put):
        client = MoDaCClient()

        mock_put.side_effect = requests.exceptions.HTTPError(
            "Bad request", response=MagicMock(status_code=400)
        )

        with self.assertRaises(requests.exceptions.HTTPError):
            client.upload_file("mock_collection", "mock_data_file.txt")

        mock_put.assert_called_once()
        mock_open.assert_called_once_with("mock_data_file.txt", "rb")


if __name__ == "__main__":
    unittest.main()
