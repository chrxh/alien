#include "NetworkResourceTreeTO.h"

bool _NetworkResourceTreeTO::isLeaf()
{
    return std::holds_alternative<BrowserLeaf>(node);
}

BrowserLeaf& _NetworkResourceTreeTO::getLeaf()
{
    return std::get<BrowserLeaf>(node);
}

BrowserFolder& _NetworkResourceTreeTO::getFolder()
{
    return std::get<BrowserFolder>(node);
}
