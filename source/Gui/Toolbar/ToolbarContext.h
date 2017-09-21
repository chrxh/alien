#pragma once
#include <QObject>

class ToolbarContext : public QObject {
	Q_OBJECT
public:
	ToolbarContext(QObject * parent = nullptr);
	~ToolbarContext();

	Q_SIGNAL void activate();
	Q_SIGNAL void deactivate();
	
};
