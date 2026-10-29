"""Dataset constants shared by training, inference and the API."""

# EuroSAT classes, in the alphabetical order used by torchvision's ImageFolder.
CLASS_NAMES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

CLASS_LABELS = {
    "AnnualCrop": "Annual crop",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous vegetation",
    "Highway": "Highway",
    "Industrial": "Industrial",
    "Pasture": "Pasture",
    "PermanentCrop": "Permanent crop",
    "Residential": "Residential",
    "River": "River",
    "SeaLake": "Sea / lake",
}

# Map colours, loosely following common land-cover map conventions.
CLASS_COLORS = {
    "AnnualCrop": "#e6c229",
    "Forest": "#1b5e20",
    "HerbaceousVegetation": "#7cb342",
    "Highway": "#757575",
    "Industrial": "#8e24aa",
    "Pasture": "#aed581",
    "PermanentCrop": "#f57c00",
    "Residential": "#e53935",
    "River": "#29b6f6",
    "SeaLake": "#0d47a1",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Sentinel-2 RGB bands have a 10 m ground sample distance; EuroSAT tiles are 64x64 px.
SENTINEL2_GSD_M = 10.0
EUROSAT_TILE_PX = 64


def label_for(class_name: str) -> str:
    """Human-readable label for a class name (falls back to the name itself)."""
    return CLASS_LABELS.get(class_name, class_name)


def color_for(class_name: str) -> str:
    """Hex colour for a class name (grey for classes outside EuroSAT)."""
    return CLASS_COLORS.get(class_name, "#9e9e9e")
