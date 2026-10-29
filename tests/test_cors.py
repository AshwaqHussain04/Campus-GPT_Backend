import re
import unittest

from starlette.middleware.cors import CORSMiddleware

from app.main import app


class CorsConfigurationTests(unittest.TestCase):
    def test_vercel_project_preview_origin_matches_cors_allowlist(self):
        cors_middleware = next(
            middleware
            for middleware in app.user_middleware
            if middleware.cls is CORSMiddleware
        )
        origin_regex = cors_middleware.kwargs["allow_origin_regex"]

        self.assertRegex(
            "https://campus-mrsxfwp9g-ashwaqhussain04s-projects.vercel.app",
            re.compile(origin_regex),
        )
        self.assertNotRegex(
            "https://unrelated-project-ashwaqhussain04s-projects.vercel.app",
            re.compile(origin_regex),
        )


if __name__ == "__main__":
    unittest.main()
