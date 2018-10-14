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
	virtual ~MainController();

	virtual void init();

	virtual void autoSave();

	virtual void onRunSimulation(bool run);
	virtual void onStepForward();
	virtual void onStepBackward(bool& emptyStack);
	virtual void onMakeSnapshot();
	virtual void onRestoreSnapshot();
	virtual void onNewSimulation(SimulationConfig const& config, double energyAtBeginning);
	virtual void onSaveSimulation(string const& filename);
	virtual bool onLoadSimulation(string const& filename);
	virtual void onRecreateSimulation(SimulationConfig const& config);
	virtual void onUpdateSimulationParametersForRunningSimulation();
	virtual void onRestrictTPS(optional<int> const& tps);

	virtual int getTimestep() const;
	virtual SimulationConfig getSimulationConfig() const;

private:
	void initSimulation(SymbolTable* symbolTable, SimulationParameters* parameters);
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

	SimulationControllerBuildFunc _controllerBuildFunc;
	SimulationAccessBuildFunc _accessBuildFunc;
};
