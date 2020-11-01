#pragma once
#include <QObject>

#include "ModelBasic/Definitions.h"

#include "Jobs.h"
#include "Definitions.h"

class MainController
	: public QObject
{
	Q_OBJECT
public:
	MainController(QObject * parent = nullptr);
	~MainController();

	void init();

    void autoSave();

	void onRunSimulation(bool run);
	void onStepForward();
	void onStepBackward(bool& emptyStack);
	void onMakeSnapshot();
	void onRestoreSnapshot();
    void onDisplayLink(bool toggled);
    void onSimulationChanger(bool toggled);
    void onNewSimulation(SimulationConfig const& config, double energyAtBeginning);
	void onSaveSimulation(string const& filename);
    enum class LoadOption { Non, SaveOldSim };
	bool onLoadSimulation(string const& filename, LoadOption option);
	void onRecreateUniverse(SimulationConfig const& config, bool extrapolateContent);
	void onUpdateSimulationParameters(SimulationParameters const& parameters);
    void onUpdateExecutionParameters(ExecutionParameters const& parameters);
    void onRestrictTPS(optional<int> const& tps);
    void onAddMostFrequentClusterToSimulation();

	int getTimestep() const;
	SimulationConfig getSimulationConfig() const;
	SimulationMonitor* getSimulationMonitor() const;

private:
	void initSimulation(SymbolTable* symbolTable, SimulationParameters const& parameters);
	void recreateSimulation(string const& serializedSimulation);
	void connectSimController() const;
	void addRandomEnergy(double amount);

    void serializeSimulationAndWaitUntilFinished();
    void autoSaveIntern(std::string const& filename);
    void saveSimulationIntern(string const& filename);

    string getPathToApp() const;


    Worker* _worker = nullptr;

	MainView* _view = nullptr;
	MainModel* _model = nullptr;

	SimulationController* _simController = nullptr;
	SimulationMonitor* _simMonitor = nullptr;

    SimulationChanger* _simChanger = nullptr;
    list<QMetaObject::Connection> _simChangerConnections;

	DataRepository* _repository = nullptr;
	Notifier* _notifier = nullptr;
	VersionController* _versionController = nullptr;
	SimulationAccess* _simAccess = nullptr;
	NumberGenerator* _numberGenerator = nullptr;
	Serializer* _serializer = nullptr;
	DescriptionHelper* _descHelper = nullptr;
    DataAnalyzer* _dataAnalyzer = nullptr;

	SimulationControllerBuildFunc _controllerBuildFunc;
	SimulationAccessBuildFunc _accessBuildFunc;

	using SimulationMonitorBuildFunc = std::function<SimulationMonitor*(SimulationController*)>;
	SimulationMonitorBuildFunc _monitorBuildFunc;

    QTimer* _autosaveTimer = nullptr;
};
