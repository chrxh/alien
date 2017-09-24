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
	QPalette palette1;
	QBrush brush2(QColor(225, 225, 225, 255));
	brush2.setStyle(Qt::SolidPattern);
	palette1.setBrush(QPalette::Active, QPalette::WindowText, brush2);
	QBrush brush3(QColor(0, 0, 0, 255));
	brush3.setStyle(Qt::SolidPattern);
	palette1.setBrush(QPalette::Active, QPalette::Button, brush3);
	QBrush brush4(QColor(0, 166, 255, 255));
	brush4.setStyle(Qt::SolidPattern);
	palette1.setBrush(QPalette::Active, QPalette::Base, brush4);
	QBrush brush5(QColor(0, 97, 145, 255));
	brush5.setStyle(Qt::SolidPattern);
	palette1.setBrush(QPalette::Active, QPalette::Window, brush5);
	palette1.setBrush(QPalette::Inactive, QPalette::WindowText, brush2);
	palette1.setBrush(QPalette::Inactive, QPalette::Button, brush3);
	palette1.setBrush(QPalette::Inactive, QPalette::Base, brush4);
	palette1.setBrush(QPalette::Inactive, QPalette::Window, brush5);
	QBrush brush6(QColor(103, 102, 100, 255));
	brush6.setStyle(Qt::SolidPattern);
	palette1.setBrush(QPalette::Disabled, QPalette::WindowText, brush6);
	palette1.setBrush(QPalette::Disabled, QPalette::Button, brush3);
	palette1.setBrush(QPalette::Disabled, QPalette::Base, brush5);
	palette1.setBrush(QPalette::Disabled, QPalette::Window, brush5);
	return palette1;
}
