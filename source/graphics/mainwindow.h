#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QByteArray>
#include <QStack>

#include "../globaldata/metadatamanager.h"

class AlienCell;
class AlienEnergy;
class AlienSimulator;
class MicroEditor;
class UniversePixelScene;
class UniverseShapeScene;
class SimulationMonitor;
class QGraphicsScene;
class QTimer;
class TutorialWindow;
namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow (AlienSimulator* simulator, MetaDataManager* meta, QWidget *parent = 0);
    ~MainWindow ();

protected slots:

    //menu: simulation
    void newSimulation ();
    void loadSimulation ();
    void saveSimulation ();
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
    void timeout ();
    void fpsForcingButtonClicked (bool toggled);
    void fpsForcingSpinboxClicked ();
    void numTokenChanged (int numToken, int maxToken, bool pasteTokenPossible);

    void incFrame ();
    void decFrame ();
    void readFrame (QDataStream& stream);

    void cellFocused (AlienCell* cell);
    void cellDefocused ();
    void energyParticleFocused (AlienEnergy* e);
    void entitiesSelected(int numCells, int numEnergyParticles);

    void updateFrameLabel ();

private:
    Ui::MainWindow *ui;

protected:
    void changeEvent(QEvent *e);

    AlienSimulator* _simulator;
    MetaDataManager* _meta;
    MicroEditor* _microEditor;

    QTimer* _timer;
    SimulationMonitor* _monitor;
    TutorialWindow* _tutorialWindow;

    int _oldFrame;
    int _frame;
    int _frameSec;

    QByteArray _serializedEnsembleData;
    QByteArray _serializedCellData;

    QStack< QByteArray > _undoUniverserses;

    QByteArray _snapshot;
};

#endif // MAINWINDOW_H
