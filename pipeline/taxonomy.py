"""Shared label schema used by hand labels, Gemini labels, and the local model."""

CATEGORIES = {
    "INFRA": "Infrastructure/utilities: roads, bridges, power, water, transit damaged or closed",
    "EVAC": "Evacuations, displacement, shelters",
    "HUMAN": "People affected: injured, dead, missing, trapped, rescued",
    "PROPERTY": "Homes/buildings flooded or damaged; water level observations on the ground",
    "NEEDS": "Requests for help or urgent needs (supplies, rescue, info)",
    "AID": "Donations, volunteering, relief efforts",
    "ADVISORY": "Warnings, official advice, closures, boil-water notices",
    "SUPPORT": "Sympathy, solidarity, prayers",
    "OTHER": "Other flood-related information or commentary",
}
CATEGORY_CODES = list(CATEGORIES)

# 0 = none, 1 = low (general info), 2 = moderate (active impact), 3 = high (life/safety, immediate)
URGENCY_LEVELS = [0, 1, 2, 3]

# Hazard the tweet is about (multi-disaster datasets). "Flood-related" = relevant and HAZARD == FLOOD.
HAZARDS = {
    "FLOOD": "Flooding of any cause: river, flash, coastal, storm surge, heavy-rain, dam or levee failure",
    "STORM": "Hurricane, typhoon, cyclone, tornado, windstorm (FLOOD if the tweet is about its flooding)",
    "QUAKE": "Earthquake or tsunami",
    "FIRE": "Wildfire, bushfire or building fire",
    "BLAST": "Explosion: industrial, refinery, gas",
    "CRASH": "Train, plane, helicopter or road crash; derailment; building collapse",
    "ATTACK": "Bombing, shooting or other violence",
    "OTHER": "Any other hazard: meteor, haze, heat, disease",
}
HAZARD_CODES = list(HAZARDS)

# CrisisLexT26 event -> hazard (used as weak hazard labels for related tweets)
EVENT_HAZARD = {
    "2012_Philipinnes_floods": "FLOOD", "2013_Alberta_floods": "FLOOD", "2013_Colorado_floods": "FLOOD",
    "2013_Manila_floods": "FLOOD", "2013_Queensland_floods": "FLOOD", "2013_Sardinia_floods": "FLOOD",
    "2012_Typhoon_Pablo": "STORM", "2013_Typhoon_Yolanda": "STORM",
    "2012_Costa_Rica_earthquake": "QUAKE", "2012_Guatemala_earthquake": "QUAKE", "2012_Italy_earthquakes": "QUAKE",
    "2013_Bohol_earthquake": "QUAKE",
    "2012_Colorado_wildfires": "FIRE", "2013_Australia_bushfire": "FIRE", "2013_Brazil_nightclub_fire": "FIRE",
    "2012_Venezuela_refinery": "BLAST", "2013_West_Texas_explosion": "BLAST",
    "2013_Glasgow_helicopter_crash": "CRASH", "2013_Lac_Megantic_train_crash": "CRASH", "2013_NY_train_crash": "CRASH",
    "2013_Spain_train_crash": "CRASH", "2013_Savar_building_collapse": "CRASH",
    "2013_Boston_bombings": "ATTACK", "2013_LA_airport_shootings": "ATTACK",
    "2013_Russia_meteor": "OTHER", "2013_Singapore_haze": "OTHER",
}
