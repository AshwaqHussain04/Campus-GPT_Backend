import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app import auth
from app.main import app as fastapi_app
from app.main import (
    LoginRequest_Admin,
    SignupRequest_Admin,
    signup_admin,
)


class AdminAuthTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(auth.settings, "admin_secret_key", "teacher-shared-secret")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_signup_rejects_an_invalid_teacher_secret(self):
        request = SignupRequest_Admin(
            name="teacher",
            email="teacher@example.com",
            password="password",
            secret_key="wrong-secret",
        )

        with patch("app.main.create_admin") as create_admin:
            with self.assertRaises(HTTPException) as error:
                signup_admin(request)

        self.assertEqual(error.exception.status_code, 403)
        create_admin.assert_not_called()

    def test_signup_creates_an_account_without_storing_the_shared_secret(self):
        request = SignupRequest_Admin(
            name="teacher",
            email="teacher@example.com",
            password="password",
            secret_key="teacher-shared-secret",
        )

        with (
            patch("app.main.create_admin") as create_admin,
            patch("app.main.hash_password", return_value="hashed-password"),
            patch("app.main.create_access_token", return_value="token"),
        ):
            response = signup_admin(request)

        create_admin.assert_called_once_with(
            "teacher",
            "hashed-password",
            "teacher@example.com",
        )
        self.assertEqual(response.access_token, "token")

    def test_login_rejects_an_invalid_teacher_secret_before_database_lookup(self):
        with patch("app.auth.get_admin_for_login") as get_admin:
            result = auth.authenticate_admin("teacher", "password", "wrong-secret")

        self.assertFalse(result)
        get_admin.assert_not_called()

    def test_login_route_passes_teacher_secret_for_authentication(self):
        request = LoginRequest_Admin(
            username="teacher",
            password="password",
            secret_key="teacher-shared-secret",
        )

        with (
            patch("app.main.authenticate_admin", return_value={"username": "teacher"}) as authenticate,
            patch("app.main.create_access_token", return_value="token"),
        ):
            login_endpoint = next(
                route.endpoint
                for route in fastapi_app.routes
                if getattr(route, "path", None) == "/auth/loginAdmin"
            )
            response = login_endpoint(request)

        authenticate.assert_called_once_with(
            "teacher",
            "password",
            "teacher-shared-secret",
        )
        self.assertEqual(response.access_token, "token")


if __name__ == "__main__":
    unittest.main()
