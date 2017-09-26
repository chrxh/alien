#pragma once

#include <QWidget>

#include "Model/Entities/Descriptions.h"
#include "Gui/Definitions.h"

class DataEditorView
	: public QObject
{
	Q_OBJECT
public:
	DataEditorView(QWidget * parent = nullptr);
	virtual ~DataEditorView() = default;

	void init(IntVector2D const& upperLeftPosition);

	void switchToNoEditor();
	void switchToClusterEditor(ClusterDescription const& cluster);
	void show(bool visible);

private:
	void update() const;

	bool _visible = false;
	DataDescription _selectedData;

	enum class EditorSelector { No, Cluster };
	EditorSelector _editorSelector = EditorSelector::No;
	
	QTabWidget* _mainTabWidget = nullptr;

	ClusterEditWidget* _clusterEditTab = nullptr;
	CellEditWidget* _cellEditTab = nullptr;
};
