#ifndef MICROEDITOR_H
#define MICROEDITOR_H

#include "../simulation/aliencellreduced.h"

#include <QWidget>
#include <QTimer>

namespace Ui {
class MicroEditor;
}

class AlienCell;
class AlienCellCluster;
class AlienEnergy;
class AlienGrid;
class CellEdit;
class ClusterEdit;
class CellComputerEdit;
class EnergyEdit;
class MetadataEdit;
class SymbolEdit;
class QTextEdit;
class QLabel;
class QTabWidget;
class QToolButton;

class MicroEditor : public QObject
{
    Q_OBJECT

public:
    explicit MicroEditor(QObject *parent = 0);
    ~MicroEditor();

    void init (QTabWidget* tabClusterWidget,
               QTabWidget* tabComputerWidget,
               QTabWidget* tabTokenWidget,
               QTabWidget* tabSymbolsWidget,
               CellEdit* cellEditor,
               ClusterEdit* clusterEditor,
               EnergyEdit* energyEditor,
               MetadataEdit* metadataEditor,
               CellComputerEdit* cellComputerEdit,
               SymbolEdit* symbolEdit,
               QTextEdit* selectionEditor,
               QToolButton* requestCellButton,
               QToolButton* requestEnergyParticleButton,
               QToolButton* delEntityButton,
               QToolButton* delClusterButton,
               QToolButton* addTokenButton,
               QToolButton* delTokenButton);
    void updateSymbolTable ();
    void updateTokenTab ();

    void setVisible (bool visible);
    bool isVisible ();
    bool eventFilter(QObject * watched, QEvent * event);

    AlienCell* getFocusedCell ();

signals:
    void requestNewCell ();                                     //to macro editor
    void requestNewEnergyParticle ();                           //to macro editor
    void updateCell (QList< AlienCell* > cells,
                     QList< AlienCellReduced > newCellsData,
                     bool clusterDataChanged);                  //to simulator
    void delSelection ();                                       //to macro editor
    void delExtendedSelection ();                                //to macro editor
    void defocus ();                                            //to macro editor
    void energyParticleUpdated (AlienEnergy* e);                //to macro editor
    void metadataUpdated ();                                    //to macro editor
    void numTokenUpdate (int numToken, int maxToken, bool pasteTokenPossible);  //to main windows

public slots:
    void computerCompilationReturn (bool error, int line);
    void defocused (bool requestDataUpdate = true);
    void cellFocused (AlienCell* cell, bool requestDataUpdate = true);
    void energyParticleFocused (AlienEnergy* e);
    void energyParticleUpdated_Slot (AlienEnergy* e);
    void reclustered (QList< AlienCellCluster* > clusters);
    void universeUpdated (AlienGrid* grid, bool force);
    void requestUpdate ();
    void entitiesSelected (int numCells, int numEnergyParticles);
    void addTokenClicked ();
    void delTokenClicked ();
    void copyTokenClicked ();
    void pasteTokenClicked ();
    void delSelectionClicked ();
    void delExtendedSelectionClicked ();

//protected:
//    void mousePressEvent(QMouseEvent * event);

private slots:
    void changesFromCellEditor (AlienCellReduced newCellProperties);
    void changesFromClusterEditor (AlienCellReduced newClusterProperties);
    void changesFromEnergyParticleEditor (QVector3D pos, QVector3D vel, qreal energyValue);
    void changesFromTokenEditor (qreal energy);
    void changesFromComputerMemoryEditor (QVector< quint8 > data);
    void changesFromTokenMemoryEditor (QVector< quint8 > data);
    void changesFromMetadataEditor (QString clusterName, QString cellName, quint8 cellColor, QString cellDescription);
    void changesFromSymbolTableEditor ();

    void clusterTabChanged (int index);
    void tokenTabChanged (int index);
    void compileButtonClicked (QString code);

private:
    void invokeUpdateCell (bool clusterDataChanged);
    void setTabSymbolsWidgetVisibility ();

    //widgets
    QTabWidget* _tabClusterWidget;
    QTabWidget* _tabComputerWidget;
    QTabWidget* _tabTokenWidget;
    QTabWidget* _tabSymbolsWidget;
    CellEdit* _cellEditor;
    ClusterEdit* _clusterEditor;
    EnergyEdit* _energyEditor;
    MetadataEdit* _metadataEditor;
    CellComputerEdit* _cellComputerEdit;
    SymbolEdit* _symbolEdit;
    QTextEdit* _selectionEditor;
    QToolButton* _requestCellButton;
    QToolButton* _requestEnergyParticleButton;
    QToolButton* _delEntityButton;
    QToolButton* _delClusterButton;
    QToolButton* _addTokenButton;
    QToolButton* _delTokenButton;

//    Ui::MicroEditor *ui;
    AlienCell* _focusCell;
    AlienCellReduced _focusCellReduced;
    AlienEnergy* _focusEnergyParticle;
    AlienGrid* _grid;
    QWidget* _tabCluster;
    QWidget* _tabCell;
    QWidget* _tabParticle;
    QWidget* _tabSelection;
    QWidget* _tabMeta;
    QWidget* _tabComputer;
    QWidget* _tabSymbolTable;
    int _currentClusterTab;
    int _currentTokenTab;

    bool _pasteTokenPossible;
    qreal _savedTokenEnergy;        //for copying tokens
    QVector< quint8 > _savedTokenData;  //for copying tokens
};


#endif // MICROEDITOR_H
