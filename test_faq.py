"""
=============================================================================
AUTOMATED TEST SUITE FOR MCA FAQ CHATBOT
=============================================================================
Validates:
1. JSON Knowledge Base format and required minimum 25 entries.
2. Question answering for Admissions, Courses, Examinations, Placements.
3. Fallback response for unsupported, speculative, and out-of-domain questions.
=============================================================================
Run with either:
    pytest test_faq.py -v
or:
    python test_faq.py
=============================================================================
"""

import os
import json
import unittest
from app import (
    load_faq_database,
    normalize_text,
    FAQRetriever,
    process_user_query,
    FALLBACK_RESPONSE,
    FAQ_FILE_PATH
)


class TestCollegeFAQChatbot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the FAQ dataset and retrieval engine once for all tests."""
        cls.faq_data = load_faq_database(FAQ_FILE_PATH)
        cls.retriever = FAQRetriever(cls.faq_data)

    # -------------------------------------------------------------------------
    # 1. KNOWLEDGE BASE INTEGRITY TESTS
    # -------------------------------------------------------------------------
    def test_faq_file_exists_and_is_valid_json(self):
        """Ensure faq_data.json exists and is parseable JSON."""
        self.assertTrue(os.path.exists(FAQ_FILE_PATH), "faq_data.json must exist in the project directory.")
        with open(FAQ_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, list, "Root JSON element must be a list of FAQ objects.")

    def test_faq_minimum_entries(self):
        """Requirement: At least 25 FAQ entries."""
        count = len(self.faq_data)
        self.assertGreaterEqual(count, 25, f"Expected at least 25 FAQ entries, but found {count}.")

    def test_faq_schema_and_categories(self):
        """Ensure all required fields exist and essential categories are covered."""
        required_keys = {"id", "category", "question", "answer", "keywords"}
        expected_categories = {"Admissions", "Courses", "Examinations", "Attendance", "Fees", "Placements", "College Facilities"}
        
        found_categories = set()
        for idx, item in enumerate(self.faq_data):
            for key in required_keys:
                self.assertIn(key, item, f"FAQ at index {idx} is missing required key '{key}'.")
            self.assertIsInstance(item["keywords"], list, f"Keywords at index {idx} must be a list.")
            self.assertGreater(len(item["keywords"]), 0, f"Keywords list at index {idx} must not be empty.")
            found_categories.add(item["category"])

        missing_categories = expected_categories - found_categories
        self.assertEqual(len(missing_categories), 0, f"Missing expected categories: {missing_categories}")

    # -------------------------------------------------------------------------
    # 2. REQUIRED 5 CORE TEST QUESTIONS
    # -------------------------------------------------------------------------
    def test_q1_admissions_question(self):
        """Test Question 1: Admissions inquiry."""
        query = "What is the eligibility criteria for admission into MCA?"
        result = process_user_query(query, self.retriever)
        
        self.assertEqual(result["status"], "SUCCESS", "Admissions query should successfully match an FAQ.")
        self.assertEqual(result["category"], "Admissions")
        self.assertIn("Bachelor's degree", result["answer"])
        self.assertGreater(result["score"], 0.22)

    def test_q2_courses_question(self):
        """Test Question 2: Courses and specializations inquiry."""
        query = "What specializations or elective subjects are offered in MCA?"
        result = process_user_query(query, self.retriever)
        
        self.assertEqual(result["status"], "SUCCESS", "Courses query should successfully match an FAQ.")
        self.assertEqual(result["category"], "Courses")
        self.assertTrue(
            "Artificial Intelligence" in result["answer"] or "electives" in result["answer"].lower(),
            "Answer should discuss departmental electives or curriculum."
        )

    def test_q3_examinations_question(self):
        """Test Question 3: Examinations and internal assessment split."""
        query = "What is the weightage of internal assessment and semester exams?"
        result = process_user_query(query, self.retriever)
        
        self.assertEqual(result["status"], "SUCCESS", "Examinations query should successfully match an FAQ.")
        self.assertEqual(result["category"], "Examinations")
        self.assertTrue(
            "40%" in result["answer"] or "internal" in result["answer"].lower(),
            "Answer should reflect examination weightage."
        )

    def test_q4_placements_question(self):
        """Test Question 4: Placements assistance inquiry."""
        query = "How does the training and placement cell assist MCA students?"
        result = process_user_query(query, self.retriever)
        
        self.assertEqual(result["status"], "SUCCESS", "Placements query should successfully match an FAQ.")
        self.assertEqual(result["category"], "Placements")
        self.assertIn("Placement Cell", result["answer"])

    def test_q5_unsupported_question_strict_fallback(self):
        """
        Test Question 5: Unsupported speculative question.
        Must NOT hallucinate or guess; must return polite fallback response.
        """
        query = "What will be the exact placement package offered to me next year?"
        result = process_user_query(query, self.retriever)
        
        self.assertEqual(
            result["status"],
            "FALLBACK",
            "Unsupported predictive query MUST trigger fallback response."
        )
        self.assertEqual(
            result["answer"],
            FALLBACK_RESPONSE,
            "Response must be the exact polite fallback text advising student to contact the department."
        )

    # -------------------------------------------------------------------------
    # 3. EXTRA OUT-OF-DOMAIN & ROBUSTNESS TESTS
    # -------------------------------------------------------------------------
    def test_typo_tolerance_admissions(self):
        """Test resilience to common student typographical errors like 'addmission'."""
        query = "addmission"
        result = process_user_query(query, self.retriever)
        self.assertEqual(result["status"], "SUCCESS", "Typo 'addmission' should resolve to Admissions FAQ.")
        self.assertEqual(result["category"], "Admissions")

    def test_out_of_domain_query(self):
        """Test completely irrelevant queries (e.g., flight booking)."""
        query = "Can you book a flight ticket to London tomorrow morning?"
        result = process_user_query(query, self.retriever)
        self.assertEqual(result["status"], "FALLBACK")
        self.assertEqual(result["answer"], FALLBACK_RESPONSE)

    def test_empty_query(self):
        """Test blank or whitespace queries."""
        query = "   "
        result = process_user_query(query, self.retriever)
        self.assertEqual(result["status"], "FALLBACK")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING MCA GENAI LAB FAQ CHATBOT AUTOMATED TESTS")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCollegeFAQChatbot)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED SUCCESSFULLY! Ready for practical viva.")
    else:
        print("\n❌ SOME TESTS FAILED. Please review output above.")
