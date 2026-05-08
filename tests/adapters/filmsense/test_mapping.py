from atomscale.adapters.filmsense.mapping import normalize_param_name


def test_psi_with_wavelength():
    p = normalize_param_name("Psi_465")
    assert p.channel_name == "psi_465"
    assert p.units == "deg"


def test_delta_with_wavelength():
    p = normalize_param_name("Delta_633")
    assert p.channel_name == "delta_633"
    assert p.units == "deg"


def test_depol_unitless():
    p = normalize_param_name("Depol_465")
    assert p.channel_name == "depol_465"
    assert p.units == ""


def test_optical_constant_n():
    p = normalize_param_name("n_465")
    assert p.channel_name == "n_465"
    assert p.units == ""


def test_optical_constant_k():
    p = normalize_param_name("k_465")
    assert p.channel_name == "k_465"
    assert p.units == ""


def test_intensity_per_wavelength():
    p = normalize_param_name("Intensity_465")
    assert p.channel_name == "intensity_465"
    assert p.units == "counts"


def test_single_layer_thickness():
    p = normalize_param_name("Thickness")
    assert p.channel_name == "thickness"
    assert p.units == "nm"


def test_multilayer_thickness_disambiguates():
    # Thickness_2 must NOT collide with the wavelength rule (no Psi/Delta/etc prefix)
    p = normalize_param_name("Thickness_2")
    assert p.channel_name == "thickness_layer_2"
    assert p.units == "nm"


def test_mse_renamed():
    p = normalize_param_name("MSE")
    assert p.channel_name == "mse_fit"


def test_aveint_renamed():
    assert normalize_param_name("AveInt").channel_name == "intensity_avg"


def test_temp_renamed():
    p = normalize_param_name("Temp")
    assert p.channel_name == "detector_temperature"
    assert p.units == "C"


def test_alignment_params():
    assert normalize_param_name("Tilt_X").channel_name == "align_tilt_x"
    assert normalize_param_name("AlignX").channel_name == "align_x"
    assert normalize_param_name("FrontZ").channel_name == "align_front_z"


def test_unknown_passthrough_lowercased():
    p = normalize_param_name("UnknownParam")
    assert p.channel_name == "unknownparam"
    assert p.units == ""


def test_whitespace_stripped():
    assert normalize_param_name(" Psi_633 ").channel_name == "psi_633"
