#include "servicelocator.h"

ServiceLocator& ServiceLocator::getInstance ()
{
    static ServiceLocator instance;
    return instance;
}
