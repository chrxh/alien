#pragma once
#include <QObject>

class ToolbarContext : public QObject {
	Q_OBJECT
public:
	ToolbarContext(QObject * parent = nullptr);
	virtual ~ToolbarContext() = default;

	Q_SIGNAL void show(bool visible);
};
