"""Local static server. Run: python serve.py; stop with Ctrl+C."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import base64
import binascii
import json
import math
import threading

import vision

ROOT = Path(__file__).resolve().parent
REFERENCE_DATA = (ROOT / 'assets' / 'rosetta-reference.png').read_bytes()
# ponytail: one CV operation at a time on this home computer; add workers if shared use needs them.
COMPUTE = threading.Lock()


def image_bytes(value):
    if not isinstance(value, str):
        raise ValueError('请选择图片。')
    try:
        result = base64.b64decode(value.split(',')[-1], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('图片数据无效，请重新选择。') from error
    if not result or len(result) > 20 * 1024 * 1024:
        raise ValueError('每张图片不能超过 20 MB。')
    return result


def image_url(image):
    return 'data:image/png;base64,' + base64.b64encode(vision.png(image)).decode('ascii')


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_head(self):
        path = unquote(urlsplit(self.path).path)
        if any(part.startswith('.') for part in path.split('/') if part):
            self.send_error(404)
            return None
        if path == '/':
            self.path = '/mobile.html'
        return super().send_head()

    def list_directory(self, path):
        self.send_error(404)
        return None

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

    def do_GET(self):
        if urlsplit(self.path).path != '/api/board':
            return super().do_GET()
        with COMPUTE:
            encoded = vision.png(vision.reference(REFERENCE_DATA)[0])
        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def json_response(self, value, status=200):
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):
        if self.path != '/api/analyze':
            self.json_response({'error': '没有这个接口。'}, 404)
            return
        try:
            if self.headers.get_content_type() != 'application/json':
                raise ValueError('请通过网页上传图片。')
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 56 * 1024 * 1024:
                raise ValueError('上传数据过大或为空。')
            self.connection.settimeout(60)
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError('图片没有上传完整，请重试。')
            request = json.loads(body)
            if not isinstance(request, dict):
                raise ValueError('请求格式不正确。')
            piece = image_bytes(request.get('piece'))
            strength = float(request.get('strength', 8))
            if not math.isfinite(strength) or not -20 <= strength <= 35:
                raise ValueError('分割强度超出范围。')
            box = request.get('box')
            if box is not None:
                if (not isinstance(box, list) or len(box) != 4 or
                    any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box) or
                    not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
                    raise ValueError('框选范围无效，请重新框选。')
                box = tuple(box)
            operation = request.get('operation', 'preprocess')
            if operation not in ('preprocess', 'match'):
                raise ValueError('操作无效。')
            with COMPUTE:
                prepared = vision.preprocess(piece, strength, box)
                output = {name: image_url(prepared[name]) for name in ('cutout', 'binary_preview', 'outlined')}
                output.update({'white_fraction': round(prepared['white_fraction'], 3), 'threshold': prepared['threshold']})
                if operation == 'match':
                    matches, message = vision.locate(prepared, REFERENCE_DATA)
                    output.update({'matches': matches, 'message': message})
                else:
                    output['message'] = '请检查轮廓和黑白文字；若截错，在原照片上框选后重新处理。'
            self.json_response(output)
        except (ValueError, TypeError, OverflowError) as error:
            self.json_response({'error': str(error)}, 400)
        except TimeoutError:
            self.json_response({'error': '上传超时，请重新选择图片。'}, 408)
        except Exception:
            import traceback
            traceback.print_exc()
            self.json_response({'error': '图片处理失败，请重新框选拼图或更换照片。'}, 500)


if __name__ == '__main__':
    try:
        with ThreadingHTTPServer(('0.0.0.0', 8765), Handler) as server:
            print('Puzzle helper: http://localhost:8765/ (Ctrl+C to stop)', flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except OSError as error:
        raise SystemExit(f'Cannot start server: {error}')
