#pragma once
#include <QObject>

#include "Model/Api/Definitions.h"

#include "Definitions.h"

class MainController
	: public QObject
{
	Q_OBJECT
public:
	MainController(QObject * parent = nullptr);
	virtual ~MainController();

	virtual void init();

	virtual void onRunSimulation(bool run);
	virtual void onStepForward();
	virtual void onStepBackward(bool& emptyStack);
	virtual void onNewSimulation(NewSimulationConfig config);
	virtual void onSaveSimulation(string const& filename);
	virtual bool onLoadSimulation(string const& filename);
	virtual void onUpdateSimulationParametersForRunningSimulation(SimulationParameters* parameters);

	virtual int getTimestep() const;

private:
	void connectSimController() const;
	void addRandomEnergy(double amount);

	//asynchronous processing
	struct SerializationOperation 
	{
		enum class Type {
			SaveToFile
		};
		Type type;
		string filename;
	};
	list<SerializationOperation> _serializationOperations;
	Q_SLOT void serializationFinished();

	MainView* _view = nullptr;
	MainModel* _model = nullptr;

	SimulationController* _simController = nullptr;
	DataController* _dataController = nullptr;
	VersionController* _versionController = nullptr;
	SimulationAccess* _simAccess = nullptr;
	Notifier* _notifier = nullptr;
	NumberGenerator* _numberGenerator = nullptr;
	Serializer* _serializer = nullptr;
	DescriptionHelper* _descHelper = nullptr;
};
