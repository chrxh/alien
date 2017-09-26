#include "Settings.h"

QFont GuiSettings::getGlobalFont ()
{
    //set font
    QFont f(GLOBAL_FONT, 8, QFont::Bold);
    f.setStyleStrategy(QFont::PreferBitmap);
    return f;
}

QFont GuiSettings::getCellFont()
{
	//set font
	QFont f(GLOBAL_FONT, 2, QFont::Normal);
	f.setStyleStrategy(QFont::PreferBitmap);
	return f;
}

QPalette GuiSettings::getPaletteForTabWidget()
{
	QPalette result;
	QBrush brush2(QColor(225, 225, 225, 255));
	brush2.setStyle(Qt::SolidPattern);
	result.setBrush(QPalette::Active, QPalette::WindowText, brush2);
	QBrush brush3(QColor(0, 0, 0, 255));
	brush3.setStyle(Qt::SolidPattern);
	result.setBrush(QPalette::Active, QPalette::Button, brush3);
	QBrush brush4(QColor(0, 166, 255, 255));
	brush4.setStyle(Qt::SolidPattern);
	result.setBrush(QPalette::Active, QPalette::Base, brush4);
	QBrush brush5(QColor(0, 97, 145, 255));
	brush5.setStyle(Qt::SolidPattern);
	result.setBrush(QPalette::Active, QPalette::Window, brush5);
	result.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
	result.setBrush(QPalette::Inactive, QPalette::Button, brush3);
	result.setBrush(QPalette::Inactive, QPalette::Base, brush4);
	result.setBrush(QPalette::Inactive, QPalette::Window, brush5);
	QBrush brush6(QColor(103, 102, 100, 255));
	brush6.setStyle(Qt::SolidPattern);
	result.setBrush(QPalette::Disabled, QPalette::WindowText, brush6);
	result.setBrush(QPalette::Disabled, QPalette::Button, brush3);
	result.setBrush(QPalette::Disabled, QPalette::Base, brush5);
	result.setBrush(QPalette::Disabled, QPalette::Window, brush5);
	return result;
}

QPalette GuiSettings::getPaletteForTab()
{
	QPalette result;
	QBrush brush(QColor(255, 255, 255, 255));
	QBrush brush3(QColor(0, 0, 0, 255));
	result.setBrush(QPalette::Active, QPalette::Text, brush);
	result.setBrush(QPalette::Active, QPalette::Base, brush3);
	QBrush brush7(QColor(152, 152, 152, 255));
	brush7.setStyle(Qt::SolidPattern);
	result.setBrush(QPalette::Active, QPalette::Window, brush7);
	result.setBrush(QPalette::Inactive, QPalette::Text, brush);
	result.setBrush(QPalette::Inactive, QPalette::Base, brush3);
	result.setBrush(QPalette::Inactive, QPalette::Window, brush7);
	QBrush brush8(QColor(120, 120, 120, 255));
	brush8.setStyle(Qt::SolidPattern);
	result.setBrush(QPalette::Disabled, QPalette::Text, brush8);
	result.setBrush(QPalette::Disabled, QPalette::Base, brush7);
	result.setBrush(QPalette::Disabled, QPalette::Window, brush7);
	return result;
}
