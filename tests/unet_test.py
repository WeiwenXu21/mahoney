from mahoney.unet import UNet


def test_unet():
    unet = UNet(1024, 2, depth=4)

    # At a depth of 4, we should have 76-ish submodules. If this drops too
    # low, that's a sign that not all submodules are being registered.
    assert len(list(unet.modules())) > 50
