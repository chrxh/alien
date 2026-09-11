#include "NameGeneratorService.h"

#include <algorithm>
#include <array>
#include <ranges>
#include <string_view>

#include <Base/Definitions.h>

#include "NumberGenerator.h"

namespace
{
    auto constexpr MaxRecentNames = 20;
    auto constexpr MaxCreationAttempts = 20;

    auto const SimulationAdjectives = std::to_array<std::string_view>({
        "Abyssal",   "Amber",     "Ancient",  "Argent",    "Ashen",     "Azure",      "Barren",   "Blazing",     "Boundless", "Brackish",
        "Bristling", "Burnished", "Cerulean", "Charred",   "Cobalt",    "Coiled",     "Crimson",  "Crystalline", "Dappled",   "Dormant",
        "Drifting",  "Drowned",   "Dusky",    "Emerald",   "Endless",   "Ephemeral",  "Eternal",  "Feral",       "Fertile",   "Flickering",
        "Floating",  "Fractured", "Frozen",   "Gilded",    "Glacial",   "Gleaming",   "Gloaming", "Hidden",      "Hollow",    "Humming",
        "Immense",   "Indigo",    "Inverted", "Iron",      "Jagged",    "Lucent",     "Luminous", "Lush",        "Marbled",   "Molten",
        "Mossy",     "Murky",     "Nascent",  "Nether",    "Nocturnal", "Obsidian",   "Onyx",     "Opaline",     "Pale",      "Peculiar",
        "Primal",    "Quiet",     "Radiant",  "Rampant",   "Restless",  "Roaring",    "Rugged",   "Rusted",      "Sable",     "Salted",
        "Scarlet",   "Scorched",  "Serene",   "Shattered", "Shifting",  "Shrouded",   "Silent",   "Slumbering",  "Solitary",  "Sunken",
        "Swelling",  "Teeming",   "Tidal",    "Tranquil",  "Twilight",  "Twisted",    "Umbral",   "Unbound",     "Vast",      "Veiled",
        "Velvet",    "Verdant",   "Violet",   "Wandering", "Weathered", "Whispering", "Wild",     "Windswept",   "Withered",  "Woven",
    });

    auto const SimulationBiotopes = std::to_array<std::string_view>({
        "Abyss",  "Archipelago", "Atoll",      "Badlands", "Barrens",   "Basin",   "Bayou",    "Bight", "Bloom",    "Bog",     "Brook",     "Canopy",
        "Canyon", "Cauldron",    "Cavern",     "Chasm",    "Cove",      "Cradle",  "Crater",   "Creek", "Crevice",  "Current", "Dell",      "Delta",
        "Depths", "Desert",      "Drift",      "Dunes",    "Estuary",   "Expanse", "Fen",      "Fjord", "Flats",    "Forest",  "Frontier",  "Garden",
        "Glade",  "Glen",        "Grotto",     "Grove",    "Gulf",      "Gully",   "Habitat",  "Haven", "Headland", "Heath",   "Highlands", "Hinterland",
        "Inlet",  "Isthmus",     "Jungle",     "Lagoon",   "Lowlands",  "Marsh",   "Meadow",   "Mire",  "Moor",     "Mouth",   "Narrows",   "Nursery",
        "Oasis",  "Outflow",     "Overgrowth", "Pass",     "Pasture",   "Plain",   "Plateau",  "Pool",  "Prairie",  "Quarry",  "Rapids",    "Ravine",
        "Reach",  "Reef",        "Ridge",      "Rise",     "Sanctuary", "Savanna", "Shallows", "Shelf", "Shoals",   "Shore",   "Sink",      "Slope",
        "Spring", "Steppe",      "Strait",     "Strand",   "Swamp",     "Terrace", "Thicket",  "Tide",  "Trench",   "Tundra",  "Vale",      "Valley",
        "Verge",  "Wastes",      "Wetland",    "Wilds",
    });

    auto const SimulationAbstractions = std::to_array<std::string_view>({
        "Ascent",    "Ashes",       "Awakening", "Balance",    "Becoming",    "Beginnings", "Cascade",    "Cinders",   "Clamor",   "Cohesion",
        "Communion", "Convergence", "Dawn",      "Decay",      "Descent",     "Discord",    "Divergence", "Dominion",  "Drifters", "Dusk",
        "Echoes",    "Ember",       "Emergence", "Endurance",  "Entropy",     "Exodus",     "Expansion",  "Fervor",    "Flux",     "Fragments",
        "Genesis",   "Harmony",     "Hunger",    "Inception",  "Inheritance", "Instinct",   "Kinship",    "Lattice",   "Legacy",   "Lineage",
        "Longing",   "Memory",      "Mirage",    "Momentum",   "Multitude",   "Onset",      "Origin",     "Overture",  "Patience", "Persistence",
        "Progeny",   "Quiescence",  "Recursion", "Reflection", "Renewal",     "Resonance",  "Rift",       "Ruin",      "Scarcity", "Silence",
        "Solitude",  "Spiral",      "Stasis",    "Succession", "Surge",       "Symbiosis",  "Symmetry",   "Threshold", "Tremor",   "Turmoil",
        "Unfolding", "Unity",       "Upheaval",  "Vestiges",   "Vigil",       "Wake",       "Whispers",   "Wonder",    "Yearning", "Zenith",
    });

    auto const GreekLetters = std::to_array<std::string_view>({
        "Alpha", "Beta", "Gamma",   "Delta", "Epsilon", "Zeta",  "Eta", "Theta",   "Iota", "Kappa", "Lambda", "Mu",
        "Nu",    "Xi",   "Omicron", "Pi",    "Rho",     "Sigma", "Tau", "Upsilon", "Phi",  "Chi",   "Psi",    "Omega",
    });

    auto const GenusPrefixes = std::to_array<std::string_view>({
        "Acantho", "Actino", "Amoebo",  "Anis",   "Arthro", "Aster", "Batho",  "Brachy", "Bryo",  "Callo",  "Campto", "Cerato", "Chaeto", "Chloro", "Chrys",
        "Cili",    "Cocco",  "Coel",    "Crypt",  "Cyan",   "Cycl",  "Cyst",   "Dendr",  "Derm",  "Desm",   "Dictyo", "Dino",   "Diplo",  "Echin",  "Ellips",
        "Erythro", "Eury",   "Flagell", "Gastro", "Glauc",  "Gono",  "Gymno",  "Halo",   "Haplo", "Helio",  "Hemi",   "Hetero", "Hexa",   "Hydro",  "Iso",
        "Lamin",   "Lepto",  "Lith",    "Lopho",  "Macro",  "Melan", "Merid",  "Meso",   "Micro", "Myco",   "Nemat",  "Noct",   "Ochro",  "Ocul",   "Oligo",
        "Ophi",    "Ortho",  "Pachy",   "Peri",   "Phyllo", "Placo", "Plano",  "Pleuro", "Polyp", "Proto",  "Pseud",  "Rhabdo", "Rhiz",   "Sarco",  "Scler",
        "Sphaer",  "Spir",   "Sten",    "Strept", "Tetra",  "Thall", "Thermo", "Trich",  "Vort",  "Xantho", "Zygo",
    });

    auto const GenusSuffixes = std::to_array<std::string_view>({
        "alis", "anthus", "aria",  "aster", "cera", "cola",  "derma", "ella",  "ensis", "ia",    "ida",  "ina",   "ion",   "ium",   "morpha", "nema",
        "odes", "oides",  "opsis", "ora",   "osa",  "phaga", "phora", "phyta", "pora",  "ptera", "soma", "spira", "stoma", "thrix", "ura",    "zoa",
    });

    auto const SpeciesEpithets = std::to_array<std::string_view>({
        "acuta",    "agilis",   "alba",     "ambigua", "amplexa",   "arcana",   "argentea",  "aspera",       "audax",     "aurea",      "avida",    "brevis",
        "caerulea", "callida",  "candida",  "capax",   "celata",    "cinerea",  "compacta",  "concava",      "crassa",    "cruenta",    "curiosa",  "densa",
        "dentata",  "dubia",    "edax",     "effusa",  "elegans",   "elongata", "errans",    "exigua",       "fallax",    "fecunda",    "ferox",    "fervida",
        "fissa",    "flava",    "fragilis", "frigida", "fugax",     "fulgens",  "furtiva",   "gelida",       "gibbosa",   "glabra",     "gracilis", "grandis",
        "hirsuta",  "horrida",  "humilis",  "ignava",  "immota",    "incerta",  "ingens",    "insatiabilis", "intrepida", "laevis",     "lenta",    "lucida",
        "lunata",   "maculata", "major",    "minor",   "mirabilis", "mobilis",  "mollis",    "mutabilis",    "nigra",     "nitida",     "nocturna", "notabilis",
        "nuda",     "obliqua",  "obscura",  "opaca",   "pallida",   "parva",    "patiens",   "perfida",      "placida",   "prolifera",  "prudens",  "pulchra",
        "radiata",  "rapax",    "robusta",  "rubra",   "sagax",     "sessilis", "solitaria", "spinosa",      "spiralis",  "symmetrica", "tenax",    "tenuis",
        "vagans",   "velox",    "viridis",  "vorax",
    });
}

namespace
{
    std::string_view pick(auto const& words)
    {
        return words.at(NumberGenerator::get().getRandomInt(toInt(words.size())));
    }

    bool isVowel(char c)
    {
        return std::string_view("aeiouy").find(c) != std::string_view::npos;
    }

    std::string createGenusName()
    {
        auto result = std::string(pick(GenusPrefixes));
        auto suffix = pick(GenusSuffixes);
        if (isVowel(result.back()) && isVowel(suffix.front())) {
            result.pop_back();
        } else if (!isVowel(result.back()) && !isVowel(suffix.front())) {
            result += 'o';
        }
        return result + std::string(suffix);
    }
}

std::string NameGeneratorService::createSimulationName()
{
    auto createName = [] {
        switch (NumberGenerator::get().getRandomInt(4)) {
        case 0:
            return std::string(pick(SimulationBiotopes)) + " of " + std::string(pick(SimulationAbstractions));
        case 1:
            return std::string(pick(GreekLetters)) + "-" + std::to_string(NumberGenerator::get().getRandomInt(1, 9)) + " "
                + std::string(pick(SimulationBiotopes));
        case 2:
            return "Project " + std::string(pick(SimulationAbstractions));
        default:
            return std::string(pick(SimulationAdjectives)) + " " + std::string(pick(SimulationBiotopes));
        }
    };
    return createUniqueName(createName, {});
}

std::string NameGeneratorService::createGenomeName(std::unordered_set<std::string> const& usedNames)
{
    auto createName = [] { return createGenusName() + " " + std::string(pick(SpeciesEpithets)); };
    return createUniqueName(createName, usedNames);
}

std::string NameGeneratorService::createUniqueName(std::function<std::string()> const& createName, std::unordered_set<std::string> const& usedNames)
{
    std::string result;
    for ([[maybe_unused]] auto attempt : std::views::iota(0, MaxCreationAttempts)) {
        result = createName();
        if (!usedNames.contains(result) && std::ranges::find(_recentNames, result) == _recentNames.end()) {
            break;
        }
    }

    _recentNames.emplace_back(result);
    if (toInt(_recentNames.size()) > MaxRecentNames) {
        _recentNames.pop_front();
    }
    return result;
}
