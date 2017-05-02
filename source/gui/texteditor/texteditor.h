#ifndef MICROEDITOR_H
#define MICROEDITOR_H

#include <QWidget>
#include <QTimer>

#include "model/entities/CellTO.h"
#include "model/Definitions.h"

namespace Ui {
class TextEditor;
}

class Cell;
class CellCluster;
class EnergyParticle;
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

class TextEditor : public QObject
{
    Q_OBJECT

public:
    TextEditor(SimulationContext* context, QObject *parent = 0);
    ~TextEditor();


	struct MicroEditorWidgets {
		QTabWidget* tabClusterWidget;
		QTabWidget* tabComputerWidget;
		QTabWidget* tabTokenWidget;
		QTabWidget* tabSymbolsWidget;
		CellEdit* cellEditor;
		ClusterEdit* clusterEditor;
		EnergyEdit* energyEditor;
		MetadataEdit* metadataEditor;
		CellComputerEdit* cellComputerEdit;
		SymbolEdit* symbolEdit;
		QTextEdit* selectionEditor;
		QToolButton* requestCellButton;
		QToolButton* requestEnergyParticleButton;
		QToolButton* delEntityButton;
		QToolButton* delClusterButton;
		QToolButton* addTokenButton;
		QToolButton* delTokenButton;
		QToolButton* buttonShowInfo;
	};
	void init(MicroEditorWidgets widgets);
	void update();

    void setVisible (bool visible);
    bool isVisible ();
    bool eventFilter(QObject * watched, QEvent * event);

    Cell* getFocusedCell ();

signals:
    void requestNewCell ();                                     //to macro editor
    void requestNewEnergyParticle ();                           //to macro editor
    void updateCell (QList< Cell* > cells,
                     QList< CellTO > newCellsData,
                     bool clusterDataChanged);                  //to simulator
    void delSelection ();                                       //to macro editor
    void delExtendedSelection ();                                //to macro editor
    void defocus ();                                            //to macro editor
    void energyParticleUpdated (EnergyParticle* e);                //to macro editor
    void metadataUpdated ();                                    //to macro editor
    void numTokenUpdate (int numToken, int maxToken, bool pasteTokenPossible);  //to main windows
	void toggleInformation(bool on);

public slots:
    void computerCompilationReturn (bool error, int line);
    void defocused (bool requestDataUpdate = true);
    void cellFocused (Cell* cell, bool requestDataUpdate = true);

	void energyParticleFocused(EnergyParticle* e);
    void energyParticleUpdated_Slot (EnergyParticle* e);
    void reclustered (QList< CellCluster* > clusters);
    void universeUpdated (SimulationContext* context, bool force);
    void requestUpdate ();

	void entitiesSelected(int numCells, int numEnergyParticles);
    void addTokenClicked ();
    void delTokenClicked ();
    void copyTokenClicked ();
    void pasteTokenClicked ();
    void delSelectionClicked ();
    void delExtendedSelectionClicked ();
	void buttonShowInfoClicked();

private slots:
    void changesFromCellEditor (CellTO newCellProperties);
    void changesFromClusterEditor (CellTO newClusterProperties);
    void changesFromEnergyParticleEditor (QVector3D pos, QVector3D vel, qreal energyValue);
    void changesFromTokenEditor (qreal energy);
    void changesFromComputerMemoryEditor (QByteArray const& data);
    void changesFromTokenMemoryEditor (QByteArray data);
    void changesFromMetadataEditor (QString clusterName, QString cellName, quint8 cellColor, QString cellDescription);
    void changesFromSymbolTableEditor ();

    void clusterTabChanged (int index);
    void tokenTabChanged (int index);
    void compileButtonClicked (QString code);

private:
	CellMetadata getCellMetadata(Cell* cell);
	CellClusterMetadata getCellClusterMetadata(Cell* cell);
	void setCellMetadata(Cell* cell, CellMetadata meta);
	void setCellClusterMetadata(Cell* cell, CellClusterMetadata meta);

    void invokeUpdateCell (bool clusterDataChanged);
    void setTabSymbolsWidgetVisibility ();

	SimulationContext* _context;

    //widgets
	MicroEditorWidgets _widgets;

    Cell* _focusCell;
    CellTO _focusCellReduced;
    EnergyParticle* _focusEnergyParticle;
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
    QByteArray _savedTokenData;  //for copying tokens
};


#endif // MICROEDITOR_H
