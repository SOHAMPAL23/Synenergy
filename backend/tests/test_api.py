"""
EnerVision AI - Backend API Tests
Tests for all FastAPI endpoints using httpx async client.
Uses in-memory SQLite for isolation (no NeonDB required for tests).
"""

import os
import sys
import uuid
import asyncio

import pytest
import pytest_asyncio

# Add project root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ─── Test fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def async_client():
    """
    Create an async test client with an in-memory SQLite database.
    Overrides DATABASE_URL before importing the app.
    """
    # Override DB to use in-memory SQLite for testing
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
    os.environ["ENVIRONMENT"] = "test"
    os.environ["DEBUG"] = "true"

    # Late import after env override
    from httpx import AsyncClient, ASGITransport
    from backend.main import app
    from backend.database.session import create_all_tables

    await create_all_tables()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture
def test_user_data():
    uid = str(uuid.uuid4())[:8]
    return {
        "email": f"test_{uid}@enervision.ai",
        "password": "TestPass123!",
        "full_name": "Test User",
        "role": "analyst",
    }


# ─── Auth Tests ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAuthEndpoints:

    async def test_register_success(self, async_client, test_user_data):
        resp = await async_client.post("/api/v1/auth/register", json=test_user_data)
        assert resp.status_code == 201
        body = resp.json()
        assert body["email"] == test_user_data["email"]
        assert body["role"] == "analyst"
        assert "id" in body
        assert "hashed_password" not in body

    async def test_register_duplicate_email_fails(self, async_client, test_user_data):
        await async_client.post("/api/v1/auth/register", json=test_user_data)
        resp = await async_client.post("/api/v1/auth/register", json=test_user_data)
        assert resp.status_code == 400

    async def test_register_invalid_role_fails(self, async_client, test_user_data):
        data = {**test_user_data, "role": "superuser", "email": "x@test.com"}
        resp = await async_client.post("/api/v1/auth/register", json=data)
        assert resp.status_code == 422

    async def test_register_admin_role_fails(self, async_client, test_user_data):
        data = {**test_user_data, "role": "admin", "email": "admin_test@test.com"}
        resp = await async_client.post("/api/v1/auth/register", json=data)
        assert resp.status_code == 400
        assert "administrator" in resp.json()["detail"]

    async def test_register_weak_password_fails(self, async_client):
        resp = await async_client.post("/api/v1/auth/register", json={
            "email": "weak@test.com",
            "password": "abc",
            "full_name": "Weak",
        })
        assert resp.status_code == 422

    async def test_login_success(self, async_client, test_user_data):
        # Register first
        uid = str(uuid.uuid4())[:8]
        user = {**test_user_data, "email": f"login_{uid}@test.com"}
        await async_client.post("/api/v1/auth/register", json=user)

        resp = await async_client.post("/api/v1/auth/login", json={
            "email": user["email"],
            "password": user["password"],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"
        assert body["expires_in"] > 0

    async def test_login_seeded_admin_success(self, async_client):
        from backend.core.config import settings
        resp = await async_client.post("/api/v1/auth/login", json={
            "email": settings.FIRST_SUPERUSER_EMAIL,
            "password": settings.FIRST_SUPERUSER_PASSWORD,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"

    async def test_login_wrong_password_fails(self, async_client, test_user_data):
        uid = str(uuid.uuid4())[:8]
        user = {**test_user_data, "email": f"wp_{uid}@test.com"}
        await async_client.post("/api/v1/auth/register", json=user)

        resp = await async_client.post("/api/v1/auth/login", json={
            "email": user["email"],
            "password": "WrongPassword!",
        })
        assert resp.status_code == 401

    async def test_login_nonexistent_user_fails(self, async_client):
        resp = await async_client.post("/api/v1/auth/login", json={
            "email": "ghost@doesnotexist.com",
            "password": "anything",
        })
        assert resp.status_code == 401

    async def test_get_me_with_valid_token(self, async_client, test_user_data):
        uid = str(uuid.uuid4())[:8]
        user = {**test_user_data, "email": f"me_{uid}@test.com"}
        await async_client.post("/api/v1/auth/register", json=user)
        login = await async_client.post("/api/v1/auth/login", json={
            "email": user["email"], "password": user["password"]
        })
        token = login.json()["access_token"]
        resp = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["email"] == user["email"]

    async def test_get_me_without_token_fails(self, async_client):
        resp = await async_client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    async def test_get_me_with_invalid_token_fails(self, async_client):
        resp = await async_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer this.is.invalid"},
        )
        assert resp.status_code == 401

    async def test_refresh_token_success(self, async_client, test_user_data):
        uid = str(uuid.uuid4())[:8]
        user = {**test_user_data, "email": f"ref_{uid}@test.com"}
        await async_client.post("/api/v1/auth/register", json=user)
        login = await async_client.post("/api/v1/auth/login", json={
            "email": user["email"], "password": user["password"]
        })
        refresh_token = login.json()["refresh_token"]
        resp = await async_client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh_token
        })
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    async def test_refresh_with_invalid_token_fails(self, async_client):
        resp = await async_client.post("/api/v1/auth/refresh", json={
            "refresh_token": "invalid.token.here"
        })
        assert resp.status_code == 401


# ─── Health Tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHealthEndpoint:

    async def test_health_returns_200(self, async_client):
        resp = await async_client.get("/api/v1/health")
        assert resp.status_code == 200

    async def test_health_body_structure(self, async_client):
        body = (await async_client.get("/api/v1/health")).json()
        assert "status" in body
        assert "version" in body
        assert "database" in body
        assert "ml_models_available" in body
        assert "timestamp" in body

    async def test_health_version_correct(self, async_client):
        body = (await async_client.get("/api/v1/health")).json()
        assert body["version"] == "1.0.0"

    async def test_root_returns_200(self, async_client):
        resp = await async_client.get("/")
        assert resp.status_code == 200
        assert "EnerVision" in resp.json()["message"]


# ─── Upload Tests ──────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def auth_headers(async_client, test_user_data):
    """Register + login and return auth headers."""
    uid = str(uuid.uuid4())[:8]
    user = {**test_user_data, "email": f"upload_{uid}@test.com"}
    await async_client.post("/api/v1/auth/register", json=user)
    login = await async_client.post("/api/v1/auth/login", json={
        "email": user["email"], "password": user["password"]
    })
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _make_csv_bytes(n: int = 300) -> bytes:
    """Generate a minimal valid energy CSV."""
    import io
    import pandas as pd
    import numpy as np

    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"DE_load_actual_entsoe_transparency": np.random.uniform(30000, 70000, n)},
        index=idx,
    )
    df.index.name = "utc_timestamp"
    buf = io.BytesIO()
    df.to_csv(buf)
    return buf.getvalue()


@pytest.mark.asyncio
class TestUploadEndpoint:

    async def test_upload_valid_csv(self, async_client, auth_headers):
        csv_data = _make_csv_bytes(300)
        resp = await async_client.post(
            "/api/v1/upload",
            files={"file": ("test_energy.csv", csv_data, "text/csv")},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["rows_valid"] > 0
        assert body["filename"] == "test_energy.csv"
        assert "upload_id" in body
        assert "time_range" in body

    async def test_upload_without_auth_fails(self, async_client):
        csv_data = _make_csv_bytes(100)
        resp = await async_client.post(
            "/api/v1/upload",
            files={"file": ("test.csv", csv_data, "text/csv")},
        )
        assert resp.status_code == 401

    async def test_upload_non_csv_rejected(self, async_client, auth_headers):
        resp = await async_client.post(
            "/api/v1/upload",
            files={"file": ("data.xlsx", b"binary data", "application/vnd.ms-excel")},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    async def test_upload_returns_upload_id(self, async_client, auth_headers):
        csv_data = _make_csv_bytes(200)
        resp = await async_client.post(
            "/api/v1/upload",
            files={"file": ("data.csv", csv_data, "text/csv")},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert len(resp.json()["upload_id"]) > 0

    async def test_upload_returns_column_list(self, async_client, auth_headers):
        csv_data = _make_csv_bytes(200)
        resp = await async_client.post(
            "/api/v1/upload",
            files={"file": ("data.csv", csv_data, "text/csv")},
            headers=auth_headers,
        )
        body = resp.json()
        assert isinstance(body["columns"], list)
        assert len(body["columns"]) > 0


# ─── ML Endpoint Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestMLEndpointsNoData:
    """ML endpoints should 404/400 when no data is uploaded."""

    async def test_forecast_without_data_returns_error(self, async_client, auth_headers):
        resp = await async_client.get("/api/v1/forecast", headers=auth_headers)
        # No data → should be 404
        assert resp.status_code in (404, 400)

    async def test_recommendations_without_data_returns_empty(self, async_client, auth_headers):
        resp = await async_client.get("/api/v1/recommendations", headers=auth_headers)
        assert resp.status_code in (200, 404, 400)

    async def test_train_without_data_returns_400(self, async_client, auth_headers):
        resp = await async_client.post("/api/v1/train", json={}, headers=auth_headers)
        assert resp.status_code in (400, 500)

    async def test_ml_endpoints_require_auth(self, async_client):
        for path in ["/api/v1/forecast", "/api/v1/recommendations"]:
            resp = await async_client.get(path)
            assert resp.status_code == 401, f"Expected 401 for {path}, got {resp.status_code}"

    async def test_dashboard_requires_auth(self, async_client):
        resp = await async_client.get("/api/v1/dashboard")
        assert resp.status_code == 401


@pytest.mark.asyncio
class TestMLEndpointsWithData:
    """ML endpoints and pipeline training with actual uploaded data."""

    async def test_full_ml_workflow_success(self, async_client, auth_headers):
        # Disable slow models and tuning during testing to keep tests fast
        from ml.utils.config_loader import config as ml_cfg
        
        models_cfg = ml_cfg.forecasting.as_dict()["models"]
        
        orig_rf_est = models_cfg["random_forest"].get("n_estimators", 100)
        orig_rf_depth = models_cfg["random_forest"].get("max_depth", 10)
        orig_xgb_enabled = models_cfg["xgboost"].get("enabled", True)
        orig_arima_enabled = models_cfg["arima"].get("enabled", True)
        orig_sarima_enabled = models_cfg["sarima"].get("enabled", True)
        orig_sarimax_enabled = models_cfg["sarimax"].get("enabled", True)

        models_cfg["random_forest"]["n_estimators"] = 5
        models_cfg["random_forest"]["max_depth"] = 3
        models_cfg["xgboost"]["enabled"] = False
        models_cfg["arima"]["enabled"] = False
        models_cfg["sarima"]["enabled"] = False
        models_cfg["sarimax"]["enabled"] = False

        try:
            # 1. Upload valid energy CSV (at least 100 rows)
            csv_data = _make_csv_bytes(300)
            upload_resp = await async_client.post(
                "/api/v1/upload",
                files={"file": ("test_energy_workflow.csv", csv_data, "text/csv")},
                headers=auth_headers,
            )
            assert upload_resp.status_code == 201
            
            # 2. Trigger training
            train_resp = await async_client.post("/api/v1/train", json={}, headers=auth_headers)
            assert train_resp.status_code == 200
            train_body = train_resp.json()
            assert train_body["status"] == "success"
            assert "best_model" in train_body
            assert "metrics" in train_body
            
            # 3. Test GET /forecast
            fc_resp = await async_client.get("/api/v1/forecast", headers=auth_headers)
            assert fc_resp.status_code == 200
            fc_body = fc_resp.json()
            assert "forecasts" in fc_body
            assert "best_model" in fc_body
            assert "24h" in fc_body["forecasts"]
            assert len(fc_body["forecasts"]["24h"]["points"]) == 24

            # 5. Test GET /recommendations
            rec_resp = await async_client.get("/api/v1/recommendations", headers=auth_headers)
            assert rec_resp.status_code == 200
            rec_body = rec_resp.json()
            assert "recommendations" in rec_body
            assert rec_body["total"] >= 0

            # 6. Test GET /explanations
            exp_resp = await async_client.get("/api/v1/explanations", headers=auth_headers)
            assert exp_resp.status_code == 200
            exp_body = exp_resp.json()
            assert "model_name" in exp_body
            assert "feature_importances" in exp_body

            # 7. Test GET /dashboard
            dash_resp = await async_client.get("/api/v1/dashboard", headers=auth_headers)
            assert dash_resp.status_code == 200
            dash_body = dash_resp.json()
            assert "stats" in dash_body
            assert "recent_forecasts" in dash_body
            assert "top_recommendations" in dash_body

            # 8. Test GET /forecast/live
            live_fc_resp = await async_client.get("/api/v1/forecast/live", headers=auth_headers)
            assert live_fc_resp.status_code == 200
            live_fc_body = live_fc_resp.json()
            assert "forecasts" in live_fc_body
            assert "best_model" in live_fc_body
            assert "24h" in live_fc_body["forecasts"]
            assert len(live_fc_body["forecasts"]["24h"]["points"]) == 24

            # 9. Test POST /predict
            from ml.models.serializer import ModelSerializer
            import pandas as pd
            ser = ModelSerializer(cfg=ml_cfg)
            metadata = ser.load_metadata(name=f"metadata_{dash_body['user']['id']}")
            assert "best_model" in metadata
            
            fe = ser.load_model(f"feature_engineer_{dash_body['user']['id']}")
            dummy_idx = pd.date_range("2020-01-01", periods=200, freq="h")
            dummy_df = pd.DataFrame({ml_cfg.data.target_column: 1.0}, index=dummy_idx)
            dummy_feat = fe.transform(dummy_df)
            feat_cols = fe.get_feature_columns(dummy_feat)
            
            mock_features = [{col: 1.0 for col in feat_cols}]
            predict_resp = await async_client.post(
                "/api/v1/predict",
                json={"features": mock_features},
                headers=auth_headers,
            )
            assert predict_resp.status_code == 200
            predict_body = predict_resp.json()
            assert "predictions" in predict_body
            assert len(predict_body["predictions"]) == 1
            assert "model_name" in predict_body
            
        finally:
            # Restore original configuration
            models_cfg["random_forest"]["n_estimators"] = orig_rf_est
            models_cfg["random_forest"]["max_depth"] = orig_rf_depth
            models_cfg["xgboost"]["enabled"] = orig_xgb_enabled
            models_cfg["arima"]["enabled"] = orig_arima_enabled
            models_cfg["sarima"]["enabled"] = orig_sarima_enabled
            models_cfg["sarimax"]["enabled"] = orig_sarimax_enabled


# ─── Rate Limit Tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRateLimiting:

    async def test_health_not_rate_limited(self, async_client):
        """Health endpoint should not be rate-limited."""
        for _ in range(10):
            resp = await async_client.get("/api/v1/health")
            assert resp.status_code == 200


# ─── Security Tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSecurityHeaders:

    async def test_request_id_in_response(self, async_client):
        resp = await async_client.get("/api/v1/health")
        assert "x-request-id" in resp.headers

    async def test_rate_limit_headers_present(self, async_client, auth_headers):
        resp = await async_client.get("/api/v1/auth/me", headers=auth_headers)
        # Rate limit headers should be present on non-health endpoints
        assert "x-ratelimit-limit" in resp.headers or resp.status_code in (401,)


# ─── Schema Validation Tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
class TestInputValidation:

    async def test_register_missing_email(self, async_client):
        resp = await async_client.post("/api/v1/auth/register", json={
            "password": "Test1234!", "full_name": "No Email"
        })
        assert resp.status_code == 422

    async def test_register_invalid_email_format(self, async_client):
        resp = await async_client.post("/api/v1/auth/register", json={
            "email": "not-an-email",
            "password": "Test1234!",
            "full_name": "Bad Email",
        })
        assert resp.status_code == 422

    async def test_login_missing_fields(self, async_client):
        resp = await async_client.post("/api/v1/auth/login", json={"email": "x@y.com"})
        assert resp.status_code == 422

    async def test_upload_empty_file_handled(self, async_client, auth_headers):
        resp = await async_client.post(
            "/api/v1/upload",
            files={"file": ("empty.csv", b"", "text/csv")},
            headers=auth_headers,
        )
        # Empty file should return an error
        assert resp.status_code in (400, 422, 500)
