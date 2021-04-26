try:
    from main import app
    import unittest

except Exception as e:
    print(f'Some modules are missing {e}')


class FlaskTest(unittest.TestCase):

    # Check if response code is 200
    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('/', content_type='html/text')
        self.assertEqual(response.status_code, 200)

    # Check if the index page renders correctly
    def test_index_page_loads(self):
        tester = app.test_client(self)
        response = tester.get('/', content_type='html/text')
        self.assertTrue(
            b'E-COV' in response.data)


if __name__ == '__main__':
    unittest.main()
