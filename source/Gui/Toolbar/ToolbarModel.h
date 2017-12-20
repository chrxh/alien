#pragma once

#include <QObject>

class ToolbarModel
	: public QObject
{
	Q_OBJECT
public:
	ToolbarModel(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~ToolbarModel() = default;

	virtual QVector2D getPositionDeltaForNewEntity();

	virtual bool isEntitySelected() const;
	virtual void setEntitySelected(bool value);

	virtual bool isEntityCopied() const;
	virtual void setEntityCopied(bool value);

	virtual bool isCellWithTokenSelected() const;
	virtual void setCellWithTokenSelected(bool value);

	virtual bool isCellWithFreeTokenSelected() const;
	virtual void setCellWithFreeTokenSelected(bool value);

	virtual bool isTokenCopied() const;
	virtual void setTokenCopied(bool value);

	virtual bool isCollectionSelected() const;
	virtual void setCollectionSelected(bool value);

	virtual bool isCollectionCopied() const;
	virtual void setCollectionCopied(bool value);

private:
	double _delta = 0.0;
	
	bool _entitySelected = false;
	bool _entityCopied = false;
	bool _cellWithTokenSelected = false;
	bool _cellWithFreeTokenSelected = false;
	bool _tokenCopied = false;
	bool _collectionSelected = false;
	bool _collectionCopied = false;
};