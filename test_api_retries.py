
import unittest
from unittest.mock import Mock, patch

import agent.runner as runner


class MockAPIError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class DataSenseAPIRetryTests(unittest.TestCase):

    def setUp(self):
        self.client = Mock()

        self.get_client_patch = patch.object(
            runner,
            "get_client",
            return_value=self.client,
        )
        self.mock_get_client = self.get_client_patch.start()

        self.sleep_patch = patch.object(
            runner.time,
            "sleep",
        )
        self.mock_sleep = self.sleep_patch.start()

        self.addCleanup(self.get_client_patch.stop)
        self.addCleanup(self.sleep_patch.stop)

    def call_api(self):
        return runner.create_chat_completion_with_retries(
            model="test-model",
            messages=[],
            tools=[],
        )

    def test_timeout_retries_and_then_succeeds(self):
        expected_response = Mock()
        expected_response.usage = None

        self.client.chat.completions.create.side_effect = [
            TimeoutError("Request timed out"),
            expected_response,
        ]

        result = self.call_api()

        self.assertIs(result, expected_response)
        self.assertEqual(
            self.client.chat.completions.create.call_count,
            2,
        )
        self.assertEqual(self.mock_sleep.call_count, 1)

    def test_rate_limit_retries(self):
        expected_response = Mock()
        expected_response.usage = None

        self.client.chat.completions.create.side_effect = [
            MockAPIError("Rate limit exceeded", 429),
            expected_response,
        ]

        result = self.call_api()

        self.assertIs(result, expected_response)
        self.assertEqual(
            self.client.chat.completions.create.call_count,
            2,
        )

    def test_server_error_retries(self):
        expected_response = Mock()
        expected_response.usage = None

        self.client.chat.completions.create.side_effect = [
            MockAPIError("Service unavailable", 503),
            expected_response,
        ]

        result = self.call_api()

        self.assertIs(result, expected_response)
        self.assertEqual(
            self.client.chat.completions.create.call_count,
            2,
        )

    def test_authentication_error_fails_without_retry(self):
        self.client.chat.completions.create.side_effect = (
            MockAPIError("Unauthorized", 401)
        )

        with self.assertRaises(MockAPIError):
            self.call_api()

        self.assertEqual(
            self.client.chat.completions.create.call_count,
            1,
        )
        self.mock_sleep.assert_not_called()

    def test_invalid_request_fails_without_retry(self):
        self.client.chat.completions.create.side_effect = (
            MockAPIError("Invalid request", 400)
        )

        with self.assertRaises(MockAPIError):
            self.call_api()

        self.assertEqual(
            self.client.chat.completions.create.call_count,
            1,
        )
        self.mock_sleep.assert_not_called()

    def test_failure_after_all_retries(self):
        self.client.chat.completions.create.side_effect = (
            TimeoutError("Request timed out")
        )

        with self.assertRaises(TimeoutError):
            self.call_api()

        self.assertEqual(
            self.client.chat.completions.create.call_count,
            runner.MAX_API_RETRIES + 1,
        )

        self.assertEqual(
            self.mock_sleep.call_count,
            runner.MAX_API_RETRIES,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
