#pragma once

#include <QWidget>

#include "Model/Api/Definitions.h"
#include "Gui/Definitions.h"

class DataEditorView
	: public QObject
{
	Q_OBJECT
public:
	DataEditorView(QWidget * parent = nullptr);
	virtual ~DataEditorView() = default;

	void init(IntVector2D const& upperLeftPosition, DataEditorModel* model, DataEditorController* controller
		, CellComputerCompiler* compiler);

	void switchToNoEditor();
	void switchToCellEditorWithComputer();
	void switchToCellEditorWithoutComputer();
	void switchToParticleEditor();
	void switchToSelectionEditor();

	void show(bool visible);
	void update() const;

private:
	enum class EditorSelector { No, CellWithComputer, CellWithoutComputer, Particle, Selection };
	void saveTabPositionForCellEditor();
	int getTabPositionForCellEditor();

	bool _visible = false;
	EditorSelector _editorSelector = EditorSelector::No;

	DataEditorModel* _model = nullptr;

	QTabWidget* _mainTabWidget = nullptr;
	ClusterEditTab* _clusterEditTab = nullptr;
	CellEditTab* _cellEditTab = nullptr;
	MetadataEditTab* _metadataEditTab = nullptr;
	ParticleEditTab* _particleEditTab = nullptr;
	SelectionEditTab* _selectionEditTab = nullptr;

	QTabWidget* _computerTabWidget = nullptr;
	CellComputerEditTab* _computerEditTab = nullptr;

	int _savedTabPosition;
};
