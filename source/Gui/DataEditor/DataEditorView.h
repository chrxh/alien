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

	void show(bool visible);
	void update() const;

private:
	enum class EditorSelector { No, CellWithComputer, CellWithoutComputer, Particle };
	void saveTabPositionForCellEditor();
	int getTabPositionForCellEditor();

	bool _visible = false;
	EditorSelector _editorSelector = EditorSelector::No;

	DataEditorModel* _model = nullptr;

	QTabWidget* _mainTabWidget = nullptr;
	ClusterEditWidget* _clusterEditTab = nullptr;
	CellEditWidget* _cellEditTab = nullptr;
	MetadataEditWidget* _metadataEditTab = nullptr;
	ParticleEditWidget* _particleEditTab = nullptr;

	QTabWidget* _computerTabWidget = nullptr;
	CellComputerEditWidget* _computerEditTab = nullptr;

	int _savedTabPosition;
};
