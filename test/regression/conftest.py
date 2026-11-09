def pytest_addoption(parser):
    parser.addoption('--regression-full', action='store_true', default=False,
                     help='check every regression fixture case, not only the quick subset (about 8 minutes)')
    parser.addoption('--regression-exact', action='store_true', default=False,
                     help='also fail when matches are found that the regression fixture does not contain')
