#ifndef MICROEDITOR_H
#define MICROEDITOR_H

#include <QWidget>
#include <QTimer>

#include "Model/Entities/CellTO.h"
#include "Model/Definitions.h"

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
    TextEditor(QObject *parent = 0);
	virtual ~TextEditor() {}


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

Q_SIGNALS:
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

public Q_SLOTS:
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

private Q_SLOTS:
    void changesFromCellEditor (CellTO newCellProperties);
    void changesFromClusterEditor (CellTO newClusterProperties);
    void changesFromEnergyParticleEditor (QVector2D pos, QVector2D vel, qreal energyValue);
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

    Cell* _focusCell = nullptr;
    CellTO _focusCellReduced;
    EnergyParticle* _focusEnergyParticle = nullptr;
    QWidget* _tabCluster = nullptr;
    QWidget* _tabCell = nullptr;
    QWidget* _tabParticle = nullptr;
    QWidget* _tabSelection = nullptr;
    QWidget* _tabMeta = nullptr;
    QWidget* _tabComputer = nullptr;
    QWidget* _tabSymbolTable = nullptr;
    int _currentClusterTab = 0;
    int _currentTokenTab = 0;

    bool _pasteTokenPossible = false;
    qreal _savedTokenEnergy = 0.0;        //for copying tokens
    QByteArray _savedTokenData;  //for copying tokens
};


#endif // MICROEDITOR_H
