import os
import unittest
from pathlib import Path

import config_data as config


class ConfigTests(unittest.TestCase):
    def test_splitter_separators_are_independent(self):
        self.assertGreater(len(config.separators), 5)
        self.assertEqual(config.separators[0], "\n\n")
        self.assertEqual(config.separators[-1], "")

    def test_runtime_paths_are_absolute(self):
        self.assertTrue(Path(config.persist_directory).is_absolute())
        self.assertTrue(Path(config.data_directory).is_absolute())
        self.assertTrue(Path(config.history_directory).is_absolute())

    def test_legacy_collection_and_fixed_session_are_not_active(self):
        self.assertNotEqual(config.collection_name, config.legacy_collection_name)
        self.assertFalse(hasattr(config, "session_config"))

    def test_retrieval_parameters_have_distinct_meanings(self):
        self.assertGreater(config.retrieval_k, 1)
        self.assertGreaterEqual(config.similarity_score_threshold, 0)
        self.assertLessEqual(config.similarity_score_threshold, 1)

    def test_dotenv_is_loaded_before_environment_values_are_resolved(self):
        self.assertEqual(os.environ["RAG_CHAT_MODEL"], config.chat_model_name)


if __name__ == "__main__":
    unittest.main()
