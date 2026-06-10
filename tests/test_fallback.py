import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import urllib.error
import json

from cad_agent.config import AgentConfig
from cad_agent.providers.gemini import GeminiProvider

class TestFallback(unittest.TestCase):
    def test_gemini_success_no_fallback(self) -> None:
        mock_response_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"result": "gemini_success"}'}
                        ]
                    }
                }
            ]
        }
        
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        
        config = AgentConfig(model="gemini-2.5-pro")
        provider = GeminiProvider(config=config)
        
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            res = provider.generate_json(
                stage="planner",
                system_prompt="Test system prompt",
                payload={"input": "test"}
            )
            self.assertEqual(res, {"result": "gemini_success"})
            self.assertEqual(mock_urlopen.call_count, 1)
            req = mock_urlopen.call_args[0][0]
            self.assertIn("generativelanguage.googleapis.com", req.full_url)

    def test_gemini_fails_no_groq_config_raises(self) -> None:
        config = AgentConfig(model="gemini-2.5-pro")
        with patch.object(AgentConfig, "groq_api_key", new_callable=PropertyMock) as mock_groq_key:
            mock_groq_key.return_value = None
            provider = GeminiProvider(config=config)
            
            with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Gemini host down")):
                with self.assertRaises(RuntimeError) as ctx:
                    provider.generate_json(
                        stage="planner",
                        system_prompt="Test system prompt",
                        payload={"input": "test"}
                    )
                self.assertIn("Gemini request failed", str(ctx.exception))

    def test_gemini_fails_groq_succeeds(self) -> None:
        config = AgentConfig(model="gemini-2.5-pro")
        
        mock_groq_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"result": "groq_fallback_success"}'
                    }
                }
            ]
        }
        
        def mock_urlopen_side_effect(req, *args, **kwargs):
            if "generativelanguage.googleapis.com" in req.full_url:
                raise urllib.error.URLError("Gemini host down")
            elif "api.groq.com" in req.full_url:
                resp = MagicMock()
                resp.read.return_value = json.dumps(mock_groq_response).encode("utf-8")
                resp.__enter__.return_value = resp
                return resp
            raise ValueError(f"Unexpected request URL: {req.full_url}")

        with patch.object(AgentConfig, "groq_api_key", new_callable=PropertyMock) as mock_groq_key:
            mock_groq_key.return_value = "fake-groq-key"
            provider = GeminiProvider(config=config)
            
            with patch("urllib.request.urlopen", side_effect=mock_urlopen_side_effect) as mock_urlopen:
                res = provider.generate_json(
                    stage="planner",
                    system_prompt="Test system prompt",
                    payload={"input": "test"}
                )
                self.assertEqual(res, {"result": "groq_fallback_success"})
                self.assertEqual(mock_urlopen.call_count, 2)

    def test_both_fail_raises_combined_error(self) -> None:
        config = AgentConfig(model="gemini-2.5-pro")
        
        def mock_urlopen_side_effect(req, *args, **kwargs):
            if "generativelanguage.googleapis.com" in req.full_url:
                raise urllib.error.URLError("Gemini failed")
            elif "api.groq.com" in req.full_url:
                raise urllib.error.URLError("Groq failed")
            raise ValueError(f"Unexpected request URL: {req.full_url}")

        with patch.object(AgentConfig, "groq_api_key", new_callable=PropertyMock) as mock_groq_key:
            mock_groq_key.return_value = "fake-groq-key"
            provider = GeminiProvider(config=config)
            
            with patch("urllib.request.urlopen", side_effect=mock_urlopen_side_effect):
                with self.assertRaises(RuntimeError) as ctx:
                    provider.generate_json(
                        stage="planner",
                        system_prompt="Test system prompt",
                        payload={"input": "test"}
                    )
                self.assertIn("Both Gemini and Groq fallback failed", str(ctx.exception))

    @patch("time.sleep")
    def test_gemini_retries_on_503_then_succeeds(self, mock_sleep) -> None:
        import io
        import urllib.error
        
        # 1st call fails with 503 HTTPError
        fp = io.BytesIO(b'{"error": {"code": 503, "status": "UNAVAILABLE", "message": "High demand"}}')
        err = urllib.error.HTTPError("url", 503, "Service Unavailable", {}, fp)
        
        # 2nd call succeeds
        mock_response_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"result": "gemini_success"}'}
                        ]
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise err
            return mock_resp
            
        config = AgentConfig(model="gemini-2.5-pro")
        provider = GeminiProvider(config=config)
        
        with patch("urllib.request.urlopen", side_effect=side_effect) as mock_urlopen:
            res = provider.generate_json(
                stage="planner",
                system_prompt="Test system prompt",
                payload={"input": "test"}
            )
            self.assertEqual(res, {"result": "gemini_success"})
            self.assertEqual(mock_urlopen.call_count, 2)
            mock_sleep.assert_called_once_with(2.0)
            
    @patch("time.sleep")
    def test_gemini_retries_on_503_then_fails_with_different_error(self, mock_sleep) -> None:
        import io
        import urllib.error
        
        # 1st call fails with 503 HTTPError
        fp1 = io.BytesIO(b'{"error": {"code": 503, "status": "UNAVAILABLE", "message": "High demand"}}')
        err1 = urllib.error.HTTPError("url", 503, "Service Unavailable", {}, fp1)
        
        # 2nd call fails with 400 Bad Request
        fp2 = io.BytesIO(b'{"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "Bad request"}}')
        err2 = urllib.error.HTTPError("url", 400, "Bad Request", {}, fp2)
        
        call_count = 0
        def side_effect(req, *args, **kwargs):
            nonlocal call_count
            # Ensure it is for Gemini
            if "generativelanguage.googleapis.com" in req.full_url:
                call_count += 1
                if call_count == 1:
                    raise err1
                raise err2
            # Handle Groq fallback if needed
            elif "api.groq.com" in req.full_url:
                raise urllib.error.URLError("Groq failed")
            raise ValueError(f"Unexpected request URL: {req.full_url}")
            
        config = AgentConfig(model="gemini-2.5-pro")
        
        with patch.object(AgentConfig, "groq_api_key", new_callable=PropertyMock) as mock_groq_key:
            mock_groq_key.return_value = "fake-groq-key"
            provider = GeminiProvider(config=config)
            
            with patch("urllib.request.urlopen", side_effect=side_effect) as mock_urlopen:
                with self.assertRaises(RuntimeError) as ctx:
                    provider.generate_json(
                        stage="planner",
                        system_prompt="Test system prompt",
                        payload={"input": "test"}
                    )
                # Ensure the failure propagated
                self.assertIn("Both Gemini and Groq fallback failed", str(ctx.exception))
                self.assertEqual(call_count, 2)
                mock_sleep.assert_called_once_with(2.0)


if __name__ == "__main__":
    unittest.main()
