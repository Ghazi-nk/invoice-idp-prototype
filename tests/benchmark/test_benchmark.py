import unittest

from unittest.mock import patch, mock_open

from app.benchmark.benchmark import calc_metrics, process_task, generate_final_results

class TestBenchmarkMetrics(unittest.TestCase):
    """Tests the metric calculation logic."""

    def test_calc_metrics(self):
        """Should calculate precision, recall, and F1-score correctly."""
        # Standard case
        self.assertEqual(calc_metrics(tp=8, fp=2, fn=2), (0.8, 0.8, 0.8))
        # Perfect score
        self.assertEqual(calc_metrics(tp=10, fp=0, fn=0), (1.0, 1.0, 1.0))

    def test_calc_metrics_division_by_zero(self):
        """Should handle division-by-zero cases gracefully."""
        # No predictions made
        self.assertEqual(calc_metrics(tp=0, fp=0, fn=5), (0.0, 0.0, 0.0))
        # No positive cases in ground truth
        self.assertEqual(calc_metrics(tp=0, fp=5, fn=0), (0.0, 0.0, 0.0))
        # No predictions and no positive cases
        self.assertEqual(calc_metrics(tp=0, fp=0, fn=0), (0.0, 0.0, 0.0))


class TestBenchmarkWorker(unittest.TestCase):
    """Tests the main worker function `process_task`."""

    @patch('app.benchmark.benchmark.check_acceptance', return_value=True)
    @patch('app.benchmark.benchmark.is_match')
    @patch('app.benchmark.benchmark.extract_invoice_fields_from_pdf')
    @patch('builtins.open', new_callable=mock_open, read_data='{"total": 100, "vendor": "TestCorp"}')
    @patch('time.perf_counter', side_effect=[10.0, 25.5])  # Start and end times
    def test_process_task_success(self, mock_time, mock_file, mock_extract, mock_is_match, mock_check_acceptance):
        """Should process a task correctly and return structured results."""
        # Arrange
        # Mock predicted data from the pipeline
        mock_extract.return_value = ({"total": 100, "vendor": "TestCorp", "iban": "DE123"}, 5.0, 10.5)

        # Mock match results: total matches, vendor does not
        def is_match_side_effect(field, gt, pred):
            return field == 'total'

        mock_is_match.side_effect = is_match_side_effect

        # Define the task arguments
        task_args = (unittest.mock.MagicMock(stem="invoice1"), "test_engine", "/path/to/label.json")

        # Act
        result = process_task(task_args, is_searchable=False)

        # Assert
        self.assertIsNotNone(result)

        # Check summary row
        summary = result['summary']
        self.assertEqual(summary['invoice'], 'invoice1')
        self.assertEqual(summary['pipeline'], 'test_engine')
        self.assertEqual(summary['total_duration'], 15.5)
        self.assertEqual(summary['ollama_duration'], 5.0)
        self.assertEqual(summary['processing_duration'], 10.5)
        self.assertEqual(summary['total'], 1)  # Matched
        self.assertEqual(summary['vendor'], 0)  # Did not match
        self.assertEqual(summary['accuracy'], 0.5)  # 1 of 2 fields correct
        
        # Calculate expected metrics
        # total: match=True, gt=100, pred=100 -> tp=1
        # vendor: match=False, gt='TestCorp', pred='TestCorp' -> fn=1 (gt exists)
        # iban (extra in pred): fp=1
        # tp=1, fp=1, fn=1
        precision, recall, f1 = calc_metrics(1, 1, 1)
        self.assertEqual(summary['precision'], precision)
        self.assertEqual(summary['recall'], recall)
        self.assertEqual(summary['f1'], f1)
        self.assertEqual(summary['acceptance'], 1)

        # Check details rows
        details = result['details']
        self.assertEqual(len(details), 2)
        self.assertEqual(details[0]['field'], 'total')
        self.assertEqual(details[0]['match'], 1)
        self.assertEqual(details[1]['field'], 'vendor')
        self.assertEqual(details[1]['match'], 0)

        # Check that check_acceptance was called with the correct data
        mock_check_acceptance.assert_called_once_with(
            {"total": 100, "vendor": "TestCorp"},  # Ground Truth
            {"total": 100, "vendor": "TestCorp", "iban": "DE123"}  # Prediction
        )

    @patch('app.benchmark.benchmark.extract_invoice_fields_from_pdf', side_effect=Exception("OCR Failed"))
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    @patch('time.perf_counter')
    def test_process_task_exception(self, mock_time, mock_file, mock_extract):
        """Should handle exceptions during processing and return None."""
        task_args = (unittest.mock.MagicMock(stem="invoice1"), "test_engine", "/path/to/label.json")
        result = process_task(task_args, is_searchable=False)
        self.assertIsNone(result)


class TestBenchmarkFinalResults(unittest.TestCase):
    """Tests the final aggregation function `generate_final_results`."""

    @patch('app.benchmark.benchmark.os.path.exists', return_value=True)
    def test_generate_final_results_success(self, mock_exists):
        """Should correctly read a summary CSV and write aggregated results."""
        # Arrange
        mock_summary_data = (
            "invoice,pipeline,accuracy,precision,recall,f1,acceptance,ollama_duration,processing_duration,total_duration\n"
            "inv1,engineA,1.0,1.0,1.0,1.0,1,5.0,5.0,10.0\n"
            "inv2,engineA,0.5,0.5,0.5,0.5,0,10.0,10.0,20.0\n"
            "inv1,engineB,0.8,0.8,0.8,0.8,1,2.0,3.0,5.0\n"
        )
        # Mock reading summary.csv and writing to results.csv
        m_open = mock_open(read_data=mock_summary_data)
        
        # Patch the OUTPUT_RESULTS_CSV path before running the test
        with patch('app.benchmark.benchmark.OUTPUT_RESULTS_CSV', 'results_ollama_model_tag.csv'):
            # Act
            with patch('builtins.open', m_open):
                generate_final_results()

            # Assert
            # Check that the results file was opened for writing
            m_open.assert_called_with('results_ollama_model_tag.csv', 'w', newline='', encoding='utf-8')

            # Get all written content
            handle = m_open()
            written_content = "".join(call.args[0] for call in handle.write.call_args_list)

            # Check header
            self.assertIn("pipeline,mean_accuracy,mean_precision,mean_recall,mean_f1,mean_ollama_duration,mean_processing_duration,mean_total_duration,acceptance_rate",
                          written_content)
            # Check aggregated data for engineA
            # accuracy: (1.0+0.5)/2=0.75, precision: (1.0+0.5)/2=0.75, total_duration: (10+20)/2=15, acceptance: (1+0)/2=0.5
            self.assertIn("engineA,0.75,0.75,0.75,0.75,7.5,7.5,15.0,0.5", written_content)
            # Check aggregated data for engineB
            self.assertIn("engineB,0.8,0.8,0.8,0.8,2.0,3.0,5.0,1.0", written_content)

    @patch('app.benchmark.benchmark.os.path.exists', return_value=False)
    @patch('builtins.print')
    def test_generate_final_results_no_summary_file(self, mock_print, mock_exists):
        """Should exit gracefully if the summary CSV does not exist."""
        generate_final_results()
        mock_print.assert_called_with("Warning: summary.csv not found. Cannot generate final results.")


if __name__ == '__main__':
    # Patch the constants to avoid dependency on a real config file
    with patch('app.benchmark.benchmark.OUTPUT_SUMMARY_CSV', 'summary.csv'), \
            patch('app.benchmark.benchmark.OUTPUT_DETAIL_CSV', 'details.csv'), \
            patch('app.benchmark.benchmark.OUTPUT_RESULTS_CSV', 'results_ollama_model_tag.csv'):
        unittest.main(argv=['first-arg-is-ignored'], exit=False)