#pragma once

#include <QWidget>

#include "Gui/Definitions.h"

class DataEditorView
	: public QObject
{
	Q_OBJECT
public:
	DataEditorView(QWidget * parent = nullptr);
	virtual ~DataEditorView() = default;

	void init(IntVector2D const& upperLeftPosition, DataEditorModel* model, DataEditorController* controller);

	void switchToNoEditor();
	void switchToCellEditorWithoutComputer();
	void switchToCellEditorWithComputer();
	void show(bool visible);
	void update() const;

private:

	bool _visible = false;
	enum class EditorSelector { No, CellWithComputer, CellWithoutComputer };
	EditorSelector _editorSelector = EditorSelector::No;

	DataEditorModel* _model = nullptr;

	QTabWidget* _mainTabWidget = nullptr;
	ClusterEditWidget* _clusterEditTab = nullptr;
	CellEditWidget* _cellEditTab = nullptr;
	MetadataEditWidget* _metadataEditTab = nullptr;

	QTabWidget* _computerTabWidget = nullptr;
	CellComputerEditWidget* _computerEditTab = nullptr;
};
