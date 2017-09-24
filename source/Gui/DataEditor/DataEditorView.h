#pragma once

#include <QWidget>

#include "Gui/Definitions.h"

class DataEditorView
	: public QObject
{
	Q_OBJECT
public:
	DataEditorView(IntVector2D const& upperLeftPosition, DataEditorModel* model, QWidget * parent = nullptr);
	virtual ~DataEditorView() = default;

	void update() const;
	void show(bool visible);

private:
	bool _visible = false;
	IntVector2D _upperLeftPosition;

	DataEditorModel* _model = nullptr;

	QTabWidget* _mainTabWidget = nullptr;

	ClusterEditor* _clusterEditor = nullptr;
};
