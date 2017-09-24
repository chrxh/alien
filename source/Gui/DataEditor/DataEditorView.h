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

	void init(IntVector2D const& upperLeftPosition);

	void update() const;
	void show(bool visible);

private:
	bool _visible = false;
	IntVector2D _upperLeftPosition;

	QTabWidget* _mainTabWidget = nullptr;

	ClusterEditor* _clusterEditor = nullptr;
};
