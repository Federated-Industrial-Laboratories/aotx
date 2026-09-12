# SPDX-License-Identifier: Apache-2.0
# Define bounded public errors without host paths or transport exception text.
class aotx_error(Exception):
    def __init__(self, status: int, message: str, code: str, param: str | None = None):
        super().__init__(message)
        self.status, self.message, self.code, self.param = status, message, code, param
        self.request_id, self.media = None, ()

    def body(self):
        kind = {401: 'authentication_error', 403: 'permission_error', 429: 'rate_limit_error'}.get(
            self.status, 'server_error' if self.status >= 500 else 'invalid_request_error')
        return {'error': {'message': self.message, 'type': kind, 'param': self.param, 'code': self.code}}


def aotx_bad(param: str, message='The request field is not supported.'):
    return aotx_error(400, message, 'invalid_field', param)
