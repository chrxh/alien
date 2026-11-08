#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <Base/StringHelper.h>

#include <Persister/Interface/SimulationLibraryService.h>

class SimulationLibraryServiceTests : public ::testing::Test
{
public:
    SimulationLibraryServiceTests()
    {
        _service = &SimulationLibraryService::get();
        auto testDirectory = std::filesystem::temp_directory_path() / "alien-simulation-library-service-tests";
        std::filesystem::remove_all(testDirectory);
        std::filesystem::create_directories(testDirectory);
        _library = _service->normalize(testDirectory);
    }

    ~SimulationLibraryServiceTests() override { std::filesystem::remove_all(_library); }

protected:
    void createFile(std::filesystem::path const& filename, std::string const& content = "content")
    {
        std::filesystem::create_directories(filename.parent_path());
        std::ofstream stream(filename, std::ios::binary);
        stream << content;
    }

    // Later files get a later time of their last change
    void createFiles(std::filesystem::path const& folder, std::vector<std::string> const& filenames)
    {
        auto lastChange = std::filesystem::file_time_type::clock::now() - std::chrono::hours(100);
        for (auto const& filename : filenames) {
            createFile(folder / filename);
            lastChange += std::chrono::hours(1);
            std::filesystem::last_write_time(folder / filename, lastChange);
        }
    }

    std::string readFile(std::filesystem::path const& filename)
    {
        std::ifstream stream(filename, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    }

    std::vector<std::string> getNames(SimulationLibraryContent const& content)
    {
        std::vector<std::string> result;
        for (auto const& simulation : content.simulations) {
            result.emplace_back(simulation.name);
        }
        return result;
    }

    SimulationFile findSimulation(std::string const& name)
    {
        for (auto const& simulation : _service->loadContent(_library).simulations) {
            if (simulation.name == name) {
                return simulation;
            }
        }
        ADD_FAILURE() << "simulation " << name << " not found";
        return {};
    }

    SimulationLibraryService* _service = nullptr;
    std::filesystem::path _library;
};

TEST_F(SimulationLibraryServiceTests, content)
{
    createFile(_library / "Aurora.sim");
    createFile(_library / "Hanging Garden" / "After 13 hours.sim");
    createFile(_library / "Hanging Garden" / "run 10" / "Start.sim");
    createFile(_library / "Hanging Garden" / "Run 9" / "Start.sim");
    createFile(_library / "Hanging Garden" / "notes.md");
    createFile(_library / ".backup" / "Hidden.sim");
    createFile(_library / "._Aurora.sim");
    createFile(_library / "Empty" / "readme.txt");

    auto content = _service->loadContent(_library);

    EXPECT_FALSE(content.truncated);
    EXPECT_EQ(
        (std::vector<std::string>{"Aurora", "Hanging Garden/After 13 hours", "Hanging Garden/Run 9/Start", "Hanging Garden/run 10/Start"}), getNames(content));

    auto const& simulation = content.simulations.at(1);
    EXPECT_EQ(_library / "Hanging Garden" / "After 13 hours.sim", simulation.path);
    EXPECT_EQ(SimulationFileKind::Named, simulation.kind);
    EXPECT_EQ(7ull, simulation.fileSize);
    EXPECT_FALSE(simulation.timestamp.empty());
    EXPECT_FALSE(simulation.timestep.has_value());
    EXPECT_FALSE(simulation.hasPicture);
    EXPECT_TRUE(simulation.description.empty());
    EXPECT_TRUE(simulation.descriptionComplete);
}

TEST_F(SimulationLibraryServiceTests, content_missingLibraryFolder)
{
    auto content = _service->loadContent(_library / "missing");

    EXPECT_TRUE(content.simulations.empty());
    EXPECT_FALSE(content.truncated);
}

TEST_F(SimulationLibraryServiceTests, content_trailingSeparator)
{
    createFile(_library / "Runs" / "Start.sim");

    auto content = _service->loadContent(_library / "Runs" / "");

    EXPECT_EQ(std::vector<std::string>{"Start"}, getNames(content));
}

TEST_F(SimulationLibraryServiceTests, content_kindFromName)
{
    createFile(_library / "Run" / "autosave_12400000.sim");
    createFile(_library / "Run" / "autosave_12400000-2.sim");
    createFile(_library / "Run" / "save_165_536_949.sim");
    createFile(_library / "Run" / "save_the_world.sim");
    createFile(_library / "Run" / "autosave_.sim");

    auto autosave = findSimulation("Run/autosave_12400000");
    EXPECT_EQ(SimulationFileKind::Autosave, autosave.kind);
    EXPECT_EQ(12400000ull, autosave.timestep);

    auto secondAutosave = findSimulation("Run/autosave_12400000-2");
    EXPECT_EQ(SimulationFileKind::Autosave, secondAutosave.kind);
    EXPECT_EQ(12400000ull, secondAutosave.timestep);

    auto savePoint = findSimulation("Run/save_165_536_949");
    EXPECT_EQ(SimulationFileKind::SavePoint, savePoint.kind);
    EXPECT_EQ(165536949ull, savePoint.timestep);

    EXPECT_EQ(SimulationFileKind::Named, findSimulation("Run/save_the_world").kind);
    EXPECT_EQ(SimulationFileKind::Named, findSimulation("Run/autosave_").kind);
}

TEST_F(SimulationLibraryServiceTests, content_simulationSavedOnExit)
{
    createFile(_library / "autosave.sim");
    createFile(_library / "Run" / "Autosave.sim");
    createFile(_library / "Run" / "autosaved.sim");

    auto savedOnExit = findSimulation("autosave");
    EXPECT_EQ(SimulationFileKind::Autosave, savedOnExit.kind);
    EXPECT_FALSE(savedOnExit.timestep.has_value());

    EXPECT_EQ(SimulationFileKind::Autosave, findSimulation("Run/Autosave").kind);
    EXPECT_EQ(SimulationFileKind::Named, findSimulation("Run/autosaved").kind);
}

TEST_F(SimulationLibraryServiceTests, content_pictureAndDescription)
{
    createFile(_library / "Aurora.sim");
    createFile(_library / "Aurora.JPG", "picture");
    createFile(_library / "Aurora.txt", "Green light\nover the mountains");
    createFile(_library / "Dawn.sim");
    createFile(_library / "Night.jpg", "picture");

    auto aurora = findSimulation("Aurora");
    EXPECT_TRUE(aurora.hasPicture);
    EXPECT_EQ("Green light\nover the mountains", aurora.description);
    EXPECT_TRUE(aurora.descriptionComplete);

    auto dawn = findSimulation("Dawn");
    EXPECT_FALSE(dawn.hasPicture);
    EXPECT_TRUE(dawn.description.empty());
}

TEST_F(SimulationLibraryServiceTests, content_descriptionWrittenByAnEditor)
{
    createFile(_library / "Aurora.sim");
    createFile(_library / "Aurora.txt", "\xef\xbb\xbf  First line\r\nSecond\tline\r\n\r\n");

    EXPECT_EQ("First line\nSecond line", findSimulation("Aurora").description);
}

TEST_F(SimulationLibraryServiceTests, content_descriptionLongerThanTheLimit)
{
    createFile(_library / "Aurora.sim");
    createFile(_library / "Aurora.txt", std::string(Const::MaxSimulationDescriptionLength + 100, 'x'));

    auto aurora = findSimulation("Aurora");

    EXPECT_EQ(static_cast<size_t>(Const::MaxSimulationDescriptionLength), aurora.description.size());
    EXPECT_FALSE(aurora.descriptionComplete);

    auto description = _service->loadDescription(_library / "Aurora.sim");
    EXPECT_EQ(aurora.description, description.text);
    EXPECT_FALSE(description.complete);
}

TEST_F(SimulationLibraryServiceTests, validSimulationNames)
{
    EXPECT_TRUE(_service->isValidSimulationName("Aurora"));
    EXPECT_TRUE(_service->isValidSimulationName("Hanging Garden/Run 15/After 13 hours"));
    EXPECT_TRUE(_service->isValidSimulationName("v1.5"));
    EXPECT_TRUE(_service->isValidSimulationName("save_the_world"));
    EXPECT_TRUE(_service->isValidSimulationName("autosave_12400000/Start"));
    EXPECT_TRUE(_service->isValidSimulationName("Autosave/Start"));
    EXPECT_TRUE(_service->isValidSimulationName("autosaved"));

    EXPECT_FALSE(_service->isValidSimulationName(""));
    EXPECT_FALSE(_service->isValidSimulationName("Hanging Garden/"));
    EXPECT_FALSE(_service->isValidSimulationName("/Aurora"));
    EXPECT_FALSE(_service->isValidSimulationName("Hanging Garden//Aurora"));
    EXPECT_FALSE(_service->isValidSimulationName("Hanging Garden\\Aurora"));
    EXPECT_FALSE(_service->isValidSimulationName("../Aurora"));
    EXPECT_FALSE(_service->isValidSimulationName("Aurora."));
    EXPECT_FALSE(_service->isValidSimulationName(" Aurora"));
    EXPECT_FALSE(_service->isValidSimulationName(".Aurora"));
    EXPECT_FALSE(_service->isValidSimulationName(".backup/Aurora"));
    EXPECT_FALSE(_service->isValidSimulationName("Aurora?"));
    EXPECT_FALSE(_service->isValidSimulationName("Runs/con"));
    EXPECT_FALSE(_service->isValidSimulationName("COM1.backup/Aurora"));
    EXPECT_FALSE(_service->isValidSimulationName(std::string(101, 'x')));
    EXPECT_FALSE(_service->isValidSimulationName("1/2/3/4/5/6/7/8/9/Aurora"));

    // Autosaves are deleted automatically, which must not happen to a simulation saved by hand
    EXPECT_FALSE(_service->isValidSimulationName("autosave_12400000"));
    EXPECT_FALSE(_service->isValidSimulationName("Hanging Garden/autosave_12400000-2"));
    EXPECT_FALSE(_service->isValidSimulationName("Hanging Garden/save_165_536_949"));

    // The simulation that is saved on exit is replaced automatically
    EXPECT_FALSE(_service->isValidSimulationName("autosave"));
    EXPECT_FALSE(_service->isValidSimulationName("Hanging Garden/Autosave"));
}

TEST_F(SimulationLibraryServiceTests, validFolderNames)
{
    EXPECT_TRUE(_service->isValidFolderName("Hanging Garden"));
    EXPECT_TRUE(_service->isValidFolderName("Hanging Garden/Run 15"));
    EXPECT_TRUE(_service->isValidFolderName("autosave_1"));
    EXPECT_TRUE(_service->isValidFolderName("Autosave"));

    EXPECT_FALSE(_service->isValidFolderName(""));
    EXPECT_FALSE(_service->isValidFolderName(".autosaves"));
    EXPECT_FALSE(_service->isValidFolderName("Hanging Garden/.git"));
    EXPECT_FALSE(_service->isValidFolderName("Hanging Garden/"));
    EXPECT_FALSE(_service->isValidFolderName("Hanging: Garden"));
    EXPECT_FALSE(_service->isValidFolderName("1/2/3/4/5/6/7/8/9"));
}

TEST_F(SimulationLibraryServiceTests, normalizeAndRelations)
{
    auto folder = _library / "Hanging Garden";
    std::filesystem::create_directories(folder);

    EXPECT_EQ(folder, _service->normalize(folder / ""));
    EXPECT_EQ(folder, _service->normalize(folder / "15" / ".."));

    EXPECT_TRUE(_service->isInside(folder, _library));
    EXPECT_TRUE(_service->isInside(_library, _library));
    EXPECT_TRUE(_service->isInside(folder / "15" / "", _library));
    EXPECT_FALSE(_service->isInside(_library, folder));
    EXPECT_FALSE(_service->isInside(_library.parent_path() / "other", _library));
}

TEST_F(SimulationLibraryServiceTests, simulationPathAndName)
{
    auto path = _service->getSimulationPath(_library, "Hanging Garden/Run 15/After 13 hours");

    EXPECT_EQ(_library / "Hanging Garden" / "Run 15" / "After 13 hours.sim", path);
    EXPECT_EQ(_library / "Hanging Garden" / "Run 15" / "After 13 hours.jpg", _service->getPicturePath(path));
    EXPECT_EQ(_library / "Hanging Garden" / "Run 15" / "After 13 hours.txt", _service->getDescriptionPath(path));
    EXPECT_EQ(_library / "v1.5.jpg", _service->getPicturePath(_service->getSimulationPath(_library, "v1.5")));

    EXPECT_EQ("Hanging Garden/Run 15/After 13 hours", _service->getSimulationName(_library, path));
    EXPECT_EQ("Aurora", _service->getSimulationName(_library, _library / "Aurora.sim"));
    EXPECT_FALSE(_service->getSimulationName(_library / "Hanging Garden", _library / "Aurora.sim").has_value());

    EXPECT_EQ("Hanging Garden/Run 15", _service->getFolderName(_library, _library / "Hanging Garden" / "Run 15" / ""));
    EXPECT_EQ("", _service->getFolderName(_library, _library));
    EXPECT_FALSE(_service->getFolderName(_library, _library.parent_path()).has_value());
}

TEST_F(SimulationLibraryServiceTests, folderSpelling)
{
    createFile(_library / "Hanging Garden" / "Run 15" / "Start.sim");
    createFile(_library / "Other" / "Start.sim");

    EXPECT_TRUE(_service->isSameFolder(_library, "Hanging Garden/Run 15", "Hanging Garden/Run 15"));
    EXPECT_FALSE(_service->isSameFolder(_library, "Hanging Garden/Run 15", "Hanging Garden"));
    EXPECT_FALSE(_service->isSameFolder(_library, "Hanging Garden", "Other"));
    EXPECT_FALSE(_service->isSameFolder(_library, "Hanging Garden", "Missing"));
    EXPECT_FALSE(_service->isSameFolder(_library, "Hanging Garden", "HANGIN~1"));
    EXPECT_FALSE(_service->changeFolderSpelling(_library, "Hanging Garden", "Other"));

    // Only some file systems take names that differ in upper and lower case for the same folder
    if (_service->isSameFolder(_library, "Hanging Garden/Run 15", "hanging garden/RUN 15")) {
        EXPECT_TRUE(_service->changeFolderSpelling(_library, "Hanging Garden/Run 15", "hanging garden/RUN 15"));
        EXPECT_EQ((std::vector<std::string>{"hanging garden/RUN 15/Start", "Other/Start"}), getNames(_service->loadContent(_library)));
    }
}

TEST_F(SimulationLibraryServiceTests, takenSimulationNames)
{
    createFile(_library / "Runs" / "Aurora.sim");
    createFile(_library / "Runs" / "Aurora.jpg");
    createFile(_library / "Runs" / "Dawn.jpg");
    createFile(_library / "Runs" / "Night.txt");

    EXPECT_TRUE(_service->isSimulationNameTaken(_library, "Runs/Aurora"));
    EXPECT_TRUE(_service->isSimulationNameTaken(_library, "Runs/Dawn"));
    EXPECT_TRUE(_service->isSimulationNameTaken(_library, "Runs/Night"));
    EXPECT_FALSE(_service->isSimulationNameTaken(_library, "Runs/Noon"));
    EXPECT_FALSE(_service->isSimulationNameTaken(_library, "Aurora"));

    EXPECT_FALSE(_service->isSimulationNameTaken(_library, "Runs/Aurora", _library / "Runs" / "Aurora.sim"));
    EXPECT_TRUE(_service->isSimulationNameTaken(_library, "Runs/Dawn", _library / "Runs" / "Aurora.sim"));
}

TEST_F(SimulationLibraryServiceTests, moveSimulation)
{
    auto source = _library / "Drafts" / "Aurora.sim";
    createFile(source, "simulation");
    createFile(_library / "Drafts" / "Aurora.jpg", "picture");
    createFile(_library / "Drafts" / "Aurora.txt", "description");

    auto target = _service->moveSimulation(_library, source, "Finished/Lights/Aurora borealis");

    ASSERT_TRUE(target.has_value());
    EXPECT_EQ(_library / "Finished" / "Lights" / "Aurora borealis.sim", *target);
    EXPECT_EQ("simulation", readFile(*target));
    EXPECT_EQ("picture", readFile(_library / "Finished" / "Lights" / "Aurora borealis.jpg"));
    EXPECT_EQ("description", readFile(_library / "Finished" / "Lights" / "Aurora borealis.txt"));
    EXPECT_FALSE(std::filesystem::exists(_library / "Drafts"));
    EXPECT_TRUE(std::filesystem::exists(_library));
}

TEST_F(SimulationLibraryServiceTests, moveSimulation_refusesTakenAndInvalidNames)
{
    auto source = _library / "Aurora.sim";
    createFile(source);
    createFile(_library / "Dawn.sim");
    createFile(_library / "Night.jpg");

    EXPECT_FALSE(_service->moveSimulation(_library, source, "Dawn").has_value());
    EXPECT_FALSE(_service->moveSimulation(_library, source, "Night").has_value());
    EXPECT_FALSE(_service->moveSimulation(_library, source, "Runs/autosave_5").has_value());
    EXPECT_FALSE(_service->moveSimulation(_library, source, "Runs/Aurora?").has_value());

    EXPECT_TRUE(std::filesystem::exists(source));
    EXPECT_FALSE(std::filesystem::exists(_library / "Runs"));
}

TEST_F(SimulationLibraryServiceTests, moveSimulation_keepsFoldersWithOtherContent)
{
    auto source = _library / "Drafts" / "Aurora.sim";
    createFile(source);
    createFile(_library / "Drafts" / "notes.md");

    ASSERT_TRUE(_service->moveSimulation(_library, source, "Aurora").has_value());

    EXPECT_TRUE(std::filesystem::exists(_library / "Drafts" / "notes.md"));
}

TEST_F(SimulationLibraryServiceTests, moveSimulation_autosaveIntoRenamedFolder)
{
    auto source = _library / "Run 15" / "autosave_12400000.sim";
    createFile(source);

    auto target = _service->moveSimulation(_library, source, "Run 16/autosave_12400000");

    ASSERT_TRUE(target.has_value());
    EXPECT_EQ(SimulationFileKind::Autosave, findSimulation("Run 16/autosave_12400000").kind);
    EXPECT_FALSE(_service->moveSimulation(_library, *target, "Run 16/autosave_5").has_value());
}

TEST_F(SimulationLibraryServiceTests, moveSimulation_changesSpellingOfFolder)
{
    auto source = _library / "Runs" / "Aurora.sim";
    createFile(source);
    createFile(_library / "Runs" / "Dawn.sim");

    // Only some file systems take names that differ in upper and lower case for the same folder
    if (!_service->isSameFolder(_library, "Runs", "RUNS")) {
        return;
    }
    auto target = _service->moveSimulation(_library, source, "RUNS/Aurora");

    ASSERT_TRUE(target.has_value());
    EXPECT_EQ((std::vector<std::string>{"RUNS/Aurora", "RUNS/Dawn"}), getNames(_service->loadContent(_library)));
}

TEST_F(SimulationLibraryServiceTests, moveSimulation_keepsOwnNameThatCouldNotBeGiven)
{
    auto ownName = std::string(120, 'x');
    auto source = _library / "Drafts" / (ownName + ".sim");
    createFile(source);

    auto target = _service->moveSimulation(_library, source, "Finished/" + ownName);

    ASSERT_TRUE(target.has_value());
    EXPECT_TRUE(std::filesystem::exists(_library / "Finished" / (ownName + ".sim")));
    EXPECT_FALSE(_service->moveSimulation(_library, *target, "Finished/" + ownName + "y").has_value());
}

TEST_F(SimulationLibraryServiceTests, deleteSimulation)
{
    auto simulationPath = _library / "Hanging Garden" / "Run 15" / "Aurora.sim";
    createFile(simulationPath);
    createFile(_library / "Hanging Garden" / "Run 15" / "Aurora.jpg");
    createFile(_library / "Hanging Garden" / "Run 15" / "Aurora.txt");
    createFile(_library / "Hanging Garden" / "Run 15" / "savepoints.json");
    createFile(_library / "Hanging Garden" / "Dawn.sim");

    _service->deleteSimulation(_library, simulationPath);

    EXPECT_FALSE(std::filesystem::exists(_library / "Hanging Garden" / "Run 15"));
    EXPECT_TRUE(std::filesystem::exists(_library / "Hanging Garden" / "Dawn.sim"));
}

TEST_F(SimulationLibraryServiceTests, deleteSimulation_keepsWhatDoesNotBelongToIt)
{
    auto folder = _library / "Run 15";
    createFile(folder / "Aurora.sim");
    createFile(folder / "Aurora.jpg");
    createFile(folder / "Aurora borealis.jpg");
    createFile(folder / "notes.md");

    _service->deleteSimulation(_library, folder / "Aurora.sim");

    EXPECT_FALSE(std::filesystem::exists(folder / "Aurora.sim"));
    EXPECT_FALSE(std::filesystem::exists(folder / "Aurora.jpg"));
    EXPECT_TRUE(std::filesystem::exists(folder / "Aurora borealis.jpg"));
    EXPECT_TRUE(std::filesystem::exists(folder / "notes.md"));
}

TEST_F(SimulationLibraryServiceTests, deleteSimulation_keepsSavePointTableOfRemainingFolder)
{
    auto folder = _library / "Run 15";
    createFile(folder / "save_100.sim");
    createFile(folder / "savepoints.json");
    createFile(folder / "Later" / "save_200.sim");

    _service->deleteSimulation(_library, folder / "save_100.sim");

    EXPECT_TRUE(std::filesystem::exists(folder / "savepoints.json"));
    EXPECT_TRUE(std::filesystem::exists(folder / "Later" / "save_200.sim"));
}

TEST_F(SimulationLibraryServiceTests, deleteSimulation_keepsLibraryFolder)
{
    createFile(_library / "Aurora.sim");

    _service->deleteSimulation(_library, _library / "Aurora.sim");

    EXPECT_TRUE(std::filesystem::is_directory(_library));
    EXPECT_TRUE(std::filesystem::is_empty(_library));
}

TEST_F(SimulationLibraryServiceTests, pictures)
{
    auto simulationPath = _library / "Aurora.sim";
    createFile(simulationPath);

    EXPECT_FALSE(_service->loadPicture(simulationPath).has_value());

    std::string jpg("\xff\xd8\x00picture", 10);
    EXPECT_TRUE(_service->savePicture(simulationPath, jpg));

    EXPECT_EQ(jpg, _service->loadPicture(simulationPath));
    EXPECT_TRUE(findSimulation("Aurora").hasPicture);
}

TEST_F(SimulationLibraryServiceTests, descriptions)
{
    auto simulationPath = _library / "Aurora.sim";
    createFile(simulationPath);

    EXPECT_TRUE(_service->loadDescription(simulationPath).text.empty());
    EXPECT_TRUE(_service->loadDescription(simulationPath).complete);

    EXPECT_TRUE(_service->saveDescription(simulationPath, "Green light"));
    EXPECT_EQ("Green light", findSimulation("Aurora").description);
    EXPECT_EQ("Green light", _service->loadDescription(simulationPath).text);

    EXPECT_TRUE(_service->saveDescription(simulationPath, "Green light\nover the mountains"));
    EXPECT_EQ("Green light\nover the mountains", findSimulation("Aurora").description);
    EXPECT_FALSE(std::filesystem::exists(_library / "Aurora.txt.tmp"));

    EXPECT_TRUE(_service->saveDescription(simulationPath, ""));
    EXPECT_FALSE(std::filesystem::exists(_library / "Aurora.txt"));
    EXPECT_TRUE(_service->saveDescription(simulationPath, ""));
}

TEST_F(SimulationLibraryServiceTests, content_genomes)
{
    createFile(_library / "Swimmers" / "Eel.genome");
    createFile(_library / "Swimmers" / "Eel.genome.jpg");
    createFile(_library / "Swimmers" / "Eel.genome.txt", "Swims in waves");
    createFile(_library / "Swimmers" / "Eel.sim");
    createFile(_library / "Swimmers" / "Eel.txt", "A pond");
    createFile(_library / "Crawler.genome");

    auto content = _service->loadContent(_library);

    ASSERT_EQ(2, content.genomes.size());
    EXPECT_EQ("Crawler", content.genomes.at(0).name);
    EXPECT_FALSE(content.genomes.at(0).hasPicture);
    EXPECT_EQ("Swimmers/Eel", content.genomes.at(1).name);
    EXPECT_EQ(_library / "Swimmers" / "Eel.genome", content.genomes.at(1).path);
    EXPECT_TRUE(content.genomes.at(1).hasPicture);
    EXPECT_EQ("Swims in waves", content.genomes.at(1).description);

    ASSERT_EQ(1, content.simulations.size());
    EXPECT_FALSE(content.simulations.front().hasPicture);
    EXPECT_EQ("A pond", content.simulations.front().description);
}

TEST_F(SimulationLibraryServiceTests, genomeNames)
{
    createFile(_library / "Eel.genome");
    createFile(_library / "Orphan.genome.txt");
    createFile(_library / "Pond.sim");

    EXPECT_TRUE(_service->isValidGenomeName("Swimmers/autosave_1"));
    EXPECT_FALSE(_service->isValidGenomeName("Swimmers/"));
    EXPECT_FALSE(_service->isValidGenomeName("Eel?"));
    EXPECT_FALSE(_service->isValidSimulationName("Eel.genome"));
    EXPECT_FALSE(_service->isValidSimulationName("Swimmers/Eel.GENOME"));
    EXPECT_TRUE(_service->isValidSimulationName("Eel.genome/Pond"));

    EXPECT_EQ(_library / "Swimmers" / "Eel.genome", _service->getGenomePath(_library, "Swimmers/Eel"));
    EXPECT_TRUE(_service->isGenomePath(_library / "Eel.GENOME"));
    EXPECT_FALSE(_service->isGenomePath(_library / "Pond.sim"));

    EXPECT_TRUE(_service->isGenomeNameTaken(_library, "Eel"));
    EXPECT_TRUE(_service->isGenomeNameTaken(_library, "Orphan"));
    EXPECT_FALSE(_service->isGenomeNameTaken(_library, "Eel", _library / "Eel.genome"));
    EXPECT_FALSE(_service->isGenomeNameTaken(_library, "Pond"));
    EXPECT_FALSE(_service->isSimulationNameTaken(_library, "Eel"));
}

TEST_F(SimulationLibraryServiceTests, moveGenome)
{
    auto genomePath = _library / "Eel.genome";
    createFile(genomePath, "genome");
    createFile(_library / "Eel.genome.jpg", "picture");
    createFile(_library / "Eel.genome.txt", "description");
    createFile(_library / "Eel.sim", "simulation");
    createFile(_library / "Eel.jpg", "picture of the simulation");
    createFile(_library / "Taken.genome");

    auto target = _service->moveGenome(_library, genomePath, "Swimmers/autosave_1");

    ASSERT_TRUE(target.has_value());
    EXPECT_EQ("genome", readFile(_library / "Swimmers" / "autosave_1.genome"));
    EXPECT_EQ("picture", readFile(_library / "Swimmers" / "autosave_1.genome.jpg"));
    EXPECT_EQ("description", readFile(_library / "Swimmers" / "autosave_1.genome.txt"));
    EXPECT_FALSE(std::filesystem::exists(genomePath));
    EXPECT_EQ("simulation", readFile(_library / "Eel.sim"));
    EXPECT_EQ("picture of the simulation", readFile(_library / "Eel.jpg"));

    EXPECT_FALSE(_service->moveGenome(_library, *target, "Taken").has_value());
    EXPECT_FALSE(_service->moveGenome(_library, *target, "Swimmers/Eel?").has_value());
}

TEST_F(SimulationLibraryServiceTests, deleteGenome)
{
    auto genomePath = _library / "Swimmers" / "Eel.genome";
    createFile(genomePath);
    createFile(_library / "Swimmers" / "Eel.genome.jpg");
    createFile(_library / "Swimmers" / "Eel.genome.txt");
    createFile(_library / "Swimmers" / "Eel.sim");
    createFile(_library / "Swimmers" / "Eel.jpg");
    createFile(_library / "Crawlers" / "Worm.genome");

    _service->deleteGenome(_library, genomePath);
    _service->deleteGenome(_library, _library / "Crawlers" / "Worm.genome");

    EXPECT_FALSE(std::filesystem::exists(genomePath));
    EXPECT_FALSE(std::filesystem::exists(_library / "Swimmers" / "Eel.genome.jpg"));
    EXPECT_FALSE(std::filesystem::exists(_library / "Swimmers" / "Eel.genome.txt"));
    EXPECT_TRUE(std::filesystem::exists(_library / "Swimmers" / "Eel.sim"));
    EXPECT_TRUE(std::filesystem::exists(_library / "Swimmers" / "Eel.jpg"));
    EXPECT_FALSE(std::filesystem::exists(_library / "Crawlers"));
}

TEST_F(SimulationLibraryServiceTests, deleteSimulation_keepsFolderWithGenomes)
{
    auto simulationPath = _library / "Swimmers" / "Pond.sim";
    createFile(simulationPath);
    createFile(_library / "Swimmers" / "Eel.genome");

    _service->deleteSimulation(_library, simulationPath);

    EXPECT_TRUE(std::filesystem::exists(_library / "Swimmers" / "Eel.genome"));
}

TEST_F(SimulationLibraryServiceTests, removeExcessAutosaves)
{
    auto folder = _library / "Run 15";
    createFiles(
        folder,
        {"autosave.sim",
         "autosave_100.sim",
         "autosave_200.sim",
         "save_150.sim",
         "autosave_300.sim",
         "Named.sim",
         "autosave_400.sim",
         "autosave_200.jpg",
         "autosave_400.jpg"});

    _service->removeExcessAutosaves(folder, 2);

    EXPECT_TRUE(std::filesystem::exists(folder / "autosave.sim"));
    EXPECT_FALSE(std::filesystem::exists(folder / "autosave_100.sim"));
    EXPECT_FALSE(std::filesystem::exists(folder / "autosave_200.sim"));
    EXPECT_FALSE(std::filesystem::exists(folder / "autosave_200.jpg"));
    EXPECT_TRUE(std::filesystem::exists(folder / "autosave_300.sim"));
    EXPECT_TRUE(std::filesystem::exists(folder / "autosave_400.sim"));
    EXPECT_TRUE(std::filesystem::exists(folder / "autosave_400.jpg"));
    EXPECT_TRUE(std::filesystem::exists(folder / "save_150.sim"));
    EXPECT_TRUE(std::filesystem::exists(folder / "Named.sim"));

    _service->removeExcessAutosaves(folder, 5);
    EXPECT_TRUE(std::filesystem::exists(folder / "autosave_300.sim"));
}
