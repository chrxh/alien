#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QByteArray>
#include <QStack>

#include "Definitions.h"
#include "model/Definitions.h"

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow (SimulationController* simulator, QWidget *parent = 0);
    ~MainWindow ();

private Q_SLOTS:

    //menu: simulation
    void newSimulation ();
    void loadSimulation ();
	void saveSimulation();
    void runClicked (bool run);
    void stepForwardClicked ();
    void stepBackClicked ();
    void snapshotUniverse ();
    void restoreUniverse ();
    void editSimulationParameters ();
    void loadSimulationParameters ();
    void saveSimulationParameters ();

    //menu: view
    void fullscreen (bool triggered);
    void setEditMode (bool editMode);
    void alienMonitorTriggered (bool on);
    void alienMonitorClosed();

    //menu: edit
    void addCell ();
    void addEnergyParticle ();
    void addRandomEnergy ();
    void copyCell ();
    void pasteCell ();
    void editSymbolTable ();
    void loadSymbols ();
    void saveSymbols ();
    void loadSymbolsWithMerging ();

    //menu: selection
    void addBlockStructure ();
    void addHexagonStructure ();
    void loadExtendedSelection ();
    void saveExtendedSelection ();
    void copyExtendedSelection ();
    void pasteExtendedSelection ();
    void multiplyRandomExtendedSelection ();
    void multiplyArrangementExtendedSelection ();

    //menu: help
    void aboutAlien ();
    void tutorialClosed();

    //misc
    void oneSecondTimeout ();
    void fpsForcingButtonClicked (bool toggled);
    void fpsForcingSpinboxClicked ();
    void numTokenChanged (int numToken, int maxToken, bool pasteTokenPossible);

    void cellFocused (Cell* cell);
    void cellDefocused ();
    void energyParticleFocused (EnergyParticle* e);
    void entitiesSelected(int numCells, int numEnergyParticles);

    void updateFrameLabel ();
    void startScreenFinished ();

private:
    Ui::MainWindow *ui;

    void changeEvent(QEvent *e);
	void stopSimulation();
	void updateControllerAndEditors();

    SimulationController* _simController;
    TextEditor* _textEditor;

    QTimer* _oneSecondTimer;
    SimulationMonitor* _monitor;
    TutorialWindow* _tutorialWindow;
    StartScreenController* _startScreen;

    QByteArray _serializedEnsembleData;
    QByteArray _serializedCellData;

    QStack< QByteArray > _undoUniverserses;

    QByteArray _snapshot;
};

#endif // MAINWINDOW_H
