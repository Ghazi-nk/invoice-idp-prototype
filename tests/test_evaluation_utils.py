
import unittest
from unittest.mock import patch, call

from app.benchmark import evaluation_utils


class TestCanonicalizationUtils(unittest.TestCase):
    """Tests for the helper functions that clean and normalize strings."""

    def test_canon_text(self):
        """Should correctly normalize, clean, and standardize text."""
        self.assertEqual(evaluation_utils.canon_text("Müller & Söhne GmbH."), "muller and sohne")
        self.assertEqual(evaluation_utils.canon_text("   Aktiengesellschaft für   Bauwesen  "), "fur bauwesen")
        self.assertEqual(evaluation_utils.canon_text("Test, Inc."), "test inc")
        self.assertEqual(evaluation_utils.canon_text(None), "")

    def test_is_name_match(self):
        """Should perform a forgiving comparison for names."""
        # Exact match after canonicalization
        self.assertTrue(evaluation_utils.is_name_match("Müller & Söhne GmbH", "müller und söhne"))
        # Predicted is a subset of expected
        self.assertTrue(evaluation_utils.is_name_match("Müller & Söhne GmbH", "Müller"))
        # Expected is a subset of predicted (should be False)
        self.assertFalse(evaluation_utils.is_name_match("Müller", "Müller & Söhne GmbH"))
        # No match
        self.assertFalse(evaluation_utils.is_name_match("Meyer AG", "Schulze GmbH"))
        # Handles None values
        self.assertTrue(evaluation_utils.is_name_match(None, None))
        self.assertFalse(evaluation_utils.is_name_match("Test", None))

    def test_canon_id(self):
        """Should correctly canonicalize ID strings."""
        self.assertEqual(evaluation_utils._canon_id("DE 123 456 789"), "DE123456789")
        self.assertEqual(evaluation_utils._canon_id("at-522-55-882"), "AT52255882")
        self.assertEqual(evaluation_utils._canon_id(None), "")


class TestMatchingLogic(unittest.TestCase):
    """Tests for the main matching and business rule functions."""

    @patch('app.benchmark.evaluation_utils.canon_text', return_value="canon_text")
    @patch('app.benchmark.evaluation_utils._canon_id', return_value="canon_id")
    @patch('app.benchmark.evaluation_utils.canon_money', return_value=123.45)
    @patch('app.benchmark.evaluation_utils.is_name_match', return_value=True)
    def test_is_match_dispatcher(self, mock_name_match, mock_money, mock_id, mock_text):
        """Should dispatch to the correct canonicalization function based on field name."""
        # Test money keys
        evaluation_utils.is_match("total_amount", "123,45", "123.45")
        mock_money.assert_called()
        mock_name_match.assert_not_called()

        # Test name keys
        mock_money.reset_mock()
        evaluation_utils.is_match("vendor_name", "Test Corp", "Test")
        mock_name_match.assert_called()
        mock_money.assert_not_called()

        # Test ID keys
        mock_name_match.reset_mock()
        evaluation_utils.is_match("iban", "DE123", "de 123")
        mock_id.assert_called()
        mock_name_match.assert_not_called()

        # Test default (other) keys
        mock_id.reset_mock()
        evaluation_utils.is_match("invoice_number", "INV-01", "inv 01")
        mock_text.assert_called()
        mock_id.assert_not_called()

    def test_is_match_null_values(self):
        """Should correctly handle None or empty strings as true values."""
        self.assertTrue(evaluation_utils.is_match("any_field", None, ""))
        self.assertTrue(evaluation_utils.is_match("any_field", "", "null"))
        self.assertTrue(evaluation_utils.is_match("any_field", "null", None))
        self.assertFalse(evaluation_utils.is_match("any_field", None, "some_value"))
        self.assertFalse(evaluation_utils.is_match("any_field", "", "some_value"))

    @patch('app.benchmark.evaluation_utils.is_match')
    def test_check_acceptance_with_po_number_success(self, mock_is_match):
        """Should return True if PO number and other required fields match."""
        # Arrange: Make all calls to is_match return True
        mock_is_match.return_value = True
        gt = {"purchase_order_number": "PO123", "recipient_name": "A", "invoice_date": "B"}
        pred = {"purchase_order_number": "PO123", "recipient_name": "A", "invoice_date": "B"}

        # Act
        result = evaluation_utils.check_acceptance(gt, pred)

        # Assert
        self.assertTrue(result)
        # Check that the correct fields were compared
        mock_is_match.assert_has_calls([
            call("purchase_order_number", "PO123", "PO123"),
            call("recipient_name", "A", "A"),
            call("invoice_date", "B", "B")
        ], any_order=True)

    @patch('app.benchmark.evaluation_utils.is_match')
    def test_check_acceptance_without_po_number_success(self, mock_is_match):
        """Should return True if no PO number but all other fields match (including one ID)."""

        # Arrange: PO number does not match, but everything else does
        def side_effect(field, gt_val, pred_val):
            return field != "purchase_order_number"

        mock_is_match.side_effect = side_effect

        gt = {"purchase_order_number": None, "iban": "DE123"}
        pred = {"purchase_order_number": "something", "iban": "DE123"}

        # Act
        result = evaluation_utils.check_acceptance(gt, pred)

        # Assert
        self.assertTrue(result)

    @patch('app.benchmark.evaluation_utils.is_match')
    def test_check_acceptance_without_po_number_fail_no_id(self, mock_is_match):
        """Should return False if no PO number and no ID fields match."""

        # Arrange: PO, IBAN, and USt-Id do not match
        def side_effect(field, gt_val, pred_val):
            return field not in ["purchase_order_number", "iban", "ust-id"]

        mock_is_match.side_effect = side_effect

        gt = {"iban": "DE123"}
        pred = {"iban": "DE456"}

        # Act
        result = evaluation_utils.check_acceptance(gt, pred)

        # Assert
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)