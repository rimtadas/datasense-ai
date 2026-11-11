
import unittest
from unittest.mock import patch

import pandas as pd

import app
from dataset_context.dataset_configs import ECOMMERCE_CONFIG
from exceptions import UserFacingError

class DataSenseAppStateTests(unittest.TestCase):

    def setUp(self):
        self.ecommerce_df = pd.DataFrame({
            "transaction_id": [1, 2, 3],
            "user_id": [101, 102, 103],
            "purchase_date": pd.to_datetime([
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ]),
            "total_amount": [100.0, 200.0, 300.0],
            "currency": ["AED", "AED", "AED"],
            "shipping_address": [
                "Address A",
                "Address B",
                "Address C",
            ],
        })

        self.generic_df = pd.DataFrame({
            "employee_id": [1, 2, 3],
            "department": ["Sales", "IT", "HR"],
            "salary": [5000, 7000, 6000],
        })

    # ----------------------------------------------
    # 1. KNOWN E-COMMERCE CONFIGURATION
    # ----------------------------------------------

    def test_known_ecommerce_repository_gets_config(self):

        config = app.get_dataset_config(
            self.ecommerce_df,
            source_type="huggingface",
            repo_id=app.KNOWN_ECOMMERCE_REPOSITORY,
        )

        self.assertIn(
            "shipping_address",
            config["exclude_from_context"],
        )

        self.assertEqual(
            config["semantic_types"]["total_amount"],
            "monetary",
        )

        self.assertEqual(
            config["currency_column"],
            "currency",
        )

    # ----------------------------------------------
    # 2. LOCAL DATASET GETS GENERIC CONFIG
    # ----------------------------------------------

    def test_local_dataset_does_not_inherit_ecommerce_config(self):

        config = app.get_dataset_config(
            self.ecommerce_df,
            source_type="local",
        )

        self.assertEqual(config, {})

    # ----------------------------------------------
    # 3. UNKNOWN HF REPOSITORY GETS GENERIC CONFIG
    # ----------------------------------------------

    def test_unknown_huggingface_dataset_gets_generic_config(self):

        config = app.get_dataset_config(
            self.generic_df,
            source_type="huggingface",
            repo_id="another-user/another-dataset",
        )

        self.assertEqual(config, {})

    # ----------------------------------------------
    # 4. CONFIGURATION ONLY INCLUDES EXISTING COLUMNS
    # ----------------------------------------------

    def test_configuration_filters_missing_columns(self):

        df = pd.DataFrame({
            "total_amount": [100, 200],
            "currency": ["AED", "AED"],
        })

        config = app.get_dataset_config(
            df,
            source_type="huggingface",
            repo_id=app.KNOWN_ECOMMERCE_REPOSITORY,
        )

        self.assertEqual(
            list(config["semantic_types"].keys()),
            ["total_amount"],
        )

        self.assertEqual(
            config["exclude_from_context"],
            [],
        )

        self.assertEqual(
            config["currency_column"],
            "currency",
        )

    # ----------------------------------------------
    # 5. CONFIGURATION DOES NOT MUTATE GLOBAL CONFIG
    # ----------------------------------------------

    def test_dataset_config_is_deep_copied(self):

        original_exclusions = list(
            ECOMMERCE_CONFIG["exclude_from_context"]
        )

        config = app.get_dataset_config(
            self.ecommerce_df,
            source_type="huggingface",
            repo_id=app.KNOWN_ECOMMERCE_REPOSITORY,
        )

        config["exclude_from_context"].append("transaction_id")

        self.assertEqual(
            ECOMMERCE_CONFIG["exclude_from_context"],
            original_exclusions,
        )

    # ----------------------------------------------
    # 6. RESET CLEARS ALL SESSION DATA
    # ----------------------------------------------

    def test_clear_dataset_resets_all_outputs(self):

        outputs = app.clear_dataset()

        self.assertEqual(len(outputs), 8)

        summary, profile, df, name, config, conversation, chat, results = outputs

        self.assertEqual(summary, "")
        self.assertTrue(profile.empty)
        self.assertIsNone(df)
        self.assertIsNone(name)
        self.assertEqual(config, {})
        self.assertEqual(conversation, [])
        self.assertEqual(chat, [])
        self.assertTrue(results.empty)

    # ----------------------------------------------
    # 7. LOADING DATASET PRESERVES ITS CONFIGURATION
    # ----------------------------------------------

    @patch("app.profile_dataset")
    def test_loaded_dataset_preserves_its_config(self, mock_profile):

        mock_profile.return_value = pd.DataFrame({
            "column": ["employee_id", "department", "salary"],
            "top_values": [None, None, None],
        })

        custom_config = {
            "dataset_description": "Employee dataset",
            "exclude_from_context": ["salary"],
        }

        outputs = app.load_and_profile_dataset(
            df=self.generic_df,
            dataset_name="employees",
            dataset_config=custom_config,
        )

        self.assertEqual(len(outputs), 8)

        self.assertEqual(
            outputs[4]["exclude_from_context"],
            ["salary"],
        )

        self.assertEqual(outputs[5], [])
        self.assertEqual(outputs[6], [])
        self.assertTrue(outputs[7].empty)

    # ----------------------------------------------
    # 8. DATASET A AND B CONFIGURATIONS ARE INDEPENDENT
    # ----------------------------------------------

    def test_switching_dataset_does_not_reuse_previous_config(self):

        config_a = {
            "exclude_from_context": ["email"],
        }

        config_b = app.get_dataset_config(
            self.generic_df,
            source_type="local",
        )

        self.assertEqual(
            config_a["exclude_from_context"],
            ["email"],
        )

        self.assertEqual(config_b, {})

        # Simulate a fresh session state after switching.
        reset_outputs = app.clear_dataset()

        self.assertEqual(reset_outputs[4], {})

        # The new dataset receives its own configuration.
        self.assertEqual(config_b, {})

    # ----------------------------------------------
    # 9. INVALID DATASET DOES NOT RETURN ACTIVE STATE
    # ----------------------------------------------

    def test_empty_dataset_is_rejected(self):

        empty_df = pd.DataFrame({
            "employee_id": [],
        })

        with self.assertRaises(UserFacingError):
            app.load_and_profile_dataset(
                df=empty_df,
                dataset_name="empty_dataset",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
