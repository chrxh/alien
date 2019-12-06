#pragma once
#include <QObject>

#include "ModelBasic/Definitions.h"

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
	void onNewSimulation(SimulationConfig const& config, double energyAtBeginning);
	void onSaveSimulation(string const& filename);
	bool onLoadSimulation(string const& filename);
	void onRecreateSimulation(SimulationConfig const& config);
	void onUpdateSimulationParametersForRunningSimulation();
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

	class _AsyncJob
	{
	public:
		virtual ~_AsyncJob() = default;

		enum class Type {
			SaveToFile,
			Recreate
		};
		Type type;

		_AsyncJob(Type type) : type(type) {}
	};
	using AsyncJob = shared_ptr<_AsyncJob>;

	class _SaveToFileJob : public _AsyncJob
	{
	public:
		virtual ~_SaveToFileJob() = default;

		string filename;

		_SaveToFileJob(string filename) : _AsyncJob(Type::SaveToFile), filename(filename) {} 
	};
	using SaveToFileJob = shared_ptr<_SaveToFileJob>;

	class _RecreateJob : public _AsyncJob
	{
	public:
		virtual ~_RecreateJob() = default;

		_RecreateJob()
			: _AsyncJob(Type::Recreate) {}
	};
	using RecreateOperation = shared_ptr<_RecreateJob>;

	list<AsyncJob> _jobsAfterSerialization;
	Q_SLOT void serializationFinished();

	MainView* _view = nullptr;
	MainModel* _model = nullptr;

	SimulationController* _simController = nullptr;
	SimulationMonitor* _simMonitor = nullptr;
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

    QTimer* _timer = nullptr;
};
