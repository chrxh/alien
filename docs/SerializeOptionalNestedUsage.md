# SerializeOptionalNested Helper Function Usage Guide

## Overview

The `serializeOptionalNested` helper function provides backward compatibility for nested complex structures in the SerializerService. It works by writing a boolean presence flag before the actual data.

## Function Signature

```cpp
template <class Archive, typename T>
void serializeOptionalNested(SerializationTask task, Archive& ar, T& data, T const& defaultValue)
```

**Parameters**:
- `task`: `SerializationTask::Save` or `SerializationTask::Load`
- `ar`: Cereal archive for serialization
- `data`: The nested structure to serialize/deserialize (passed by reference)
- `defaultValue`: Default value to use if data is not present in the stream

## When to Use

Use `serializeOptionalNested` for:
1. **Variant types** containing nested genome/cell descriptions
2. **Vector collections** of complex structures
3. **Nested objects** that might be added in future versions
4. **Any structure** serialized with `ar()` that needs backward compatibility

## Usage Examples

### Example 1: Variant Field (Most Common)

```cpp
template <class Archive>
void loadSave(SerializationTask task, Archive& ar, SensorGenomeDescription& data)
{
    SensorGenomeDescription defaultObject;
    auto auxiliaries = getLoadSaveMap(task, ar);
    
    // Regular fields use LoadSave map
    loadSave(task, auxiliaries, Id_SensorGenome_AutoTriggerInterval, 
             data._autoTriggerInterval, defaultObject._autoTriggerInterval);
    loadSave(task, auxiliaries, Id_SensorGenome_MinRange, 
             data._minRange, defaultObject._minRange);
    loadSave(task, auxiliaries, Id_SensorGenome_MaxRange, 
             data._maxRange, defaultObject._maxRange);
    
    processLoadSaveMap(task, ar, auxiliaries);
    
    // BEFORE (no backward compatibility):
    // ar(data._mode);
    
    // AFTER (with backward compatibility):
    serializeOptionalNested(task, ar, data._mode, defaultObject._mode);
}
```

### Example 2: Multiple Nested Objects

```cpp
template <class Archive>
void loadSave(SerializationTask task, Archive& ar, CellDescription& data)
{
    CellDescription defaultObject;
    auto auxiliaries = getLoadSaveMap(task, ar);
    
    // ... LoadSave map calls for primitive fields ...
    
    processLoadSaveMap(task, ar, auxiliaries);
    
    // BEFORE:
    // ar(data._connections, data._cellType, data._signal, 
    //    data._signalRestriction, data._neuralNetwork);
    
    // AFTER (each can independently use defaults):
    serializeOptionalNested(task, ar, data._connections, defaultObject._connections);
    serializeOptionalNested(task, ar, data._cellType, defaultObject._cellType);
    serializeOptionalNested(task, ar, data._signal, defaultObject._signal);
    serializeOptionalNested(task, ar, data._signalRestriction, defaultObject._signalRestriction);
    serializeOptionalNested(task, ar, data._neuralNetwork, defaultObject._neuralNetwork);
}
```

### Example 3: Vector of Structures

```cpp
template <class Archive>
void loadSave(SerializationTask task, Archive& ar, GeneDescription& data)
{
    GeneDescription defaultObject;
    auto auxiliaries = getLoadSaveMap(task, ar);
    
    // ... LoadSave map calls ...
    
    processLoadSaveMap(task, ar, auxiliaries);
    
    // BEFORE:
    // ar(data._nodes);
    
    // AFTER:
    serializeOptionalNested(task, ar, data._nodes, defaultObject._nodes);
}
```

## How It Works

### During Save (SerializationTask::Save)

1. Writes `true` flag to indicate data is present
2. Serializes the actual data

```
Stream: [true][data...]
```

### During Load (SerializationTask::Load)

1. Reads the presence flag
2. If `true`: Deserializes the data
3. If `false`: Uses the provided default value

This allows:
- **Backward compatibility**: Old files without the flag won't have it, but new code can handle missing nested data
- **Forward compatibility**: New files with optional data can be read by old code (if combined with version checking)

## Migration Strategy

### For New Code

Always use `serializeOptionalNested` for new nested structures:

```cpp
// ✓ GOOD - New field with backward compatibility
serializeOptionalNested(task, ar, data._newComplexField, defaultObject._newComplexField);
```

### For Existing Code

**Option A: Gradual Migration (Recommended)**

Migrate when structures need updates, maintaining compatibility:

```cpp
// Check if we're loading from an old version
if (task == SerializationTask::Load) {
    // Try new format with presence flag
    try {
        serializeOptionalNested(task, ar, data._mode, defaultObject._mode);
    } catch (...) {
        // Fall back to old direct serialization for old files
        ar(data._mode);
    }
} else {
    // Always save with new format
    serializeOptionalNested(task, ar, data._mode, defaultObject._mode);
}
```

**Option B: Version-Based Migration**

Use version numbers to handle transition:

```cpp
if (version >= SERIALIZATION_VERSION_WITH_OPTIONAL_NESTED) {
    serializeOptionalNested(task, ar, data._mode, defaultObject._mode);
} else {
    ar(data._mode);
}
```

## Benefits

1. **Backward Compatibility**: Load old save files without errors
2. **Default Values**: Missing nested data gets sensible defaults
3. **Incremental Adoption**: Can be applied selectively
4. **No Exceptions**: Clean control flow without try-catch
5. **Type Safe**: Compiler enforces correct types

## Limitations

1. **Format Change**: Adds a boolean flag (1 byte overhead per nested structure)
2. **Not Retroactive**: Doesn't help with existing serialized data unless migrated
3. **Whole Structure**: Applies to entire nested object, not individual fields within it
4. **Manual Application**: Requires updating code for each nested structure

## Comparison with Alternatives

| Approach | Compatibility | Complexity | Overhead | Granularity |
|----------|--------------|------------|----------|-------------|
| `serializeOptionalNested` | Good | Low | 1 byte/field | Structure-level |
| Recursive LoadSave Maps | Excellent | Very High | Moderate | Field-level |
| Cereal Versioning | Good | Medium | None | Structure-level |
| Try-Catch Fallback | Fair | Low | Variable | Structure-level |

## Best Practices

1. **Always provide defaults**: Use `defaultObject._field` pattern consistently
2. **Document changes**: Comment when migrating from `ar()` to `serializeOptionalNested`
3. **Test backward compatibility**: Keep old save files for regression testing
4. **Consider overhead**: For very large collections, evaluate serialization size impact
5. **Combine with versioning**: Use alongside version numbers for maximum flexibility

## Example: Complete Migration

### Before Migration

```cpp
template <class Archive>
void loadSave(SerializationTask task, Archive& ar, NodeDescription& data)
{
    NodeDescription defaultObject;
    auto auxiliaries = getLoadSaveMap(task, ar);
    loadSave(task, auxiliaries, Id_Node_ReferenceAngle, data._referenceAngle, defaultObject._referenceAngle);
    loadSave(task, auxiliaries, Id_Node_Color, data._color, defaultObject._color);
    loadSave(task, auxiliaries, Id_Node_NumAdditionalConnections, data._numAdditionalConnections, defaultObject._numAdditionalConnections);
    processLoadSaveMap(task, ar, auxiliaries);
    
    ar(data._neuralNetwork, data._cellType, data._signalRestriction);
}
```

### After Migration

```cpp
template <class Archive>
void loadSave(SerializationTask task, Archive& ar, NodeDescription& data)
{
    NodeDescription defaultObject;
    auto auxiliaries = getLoadSaveMap(task, ar);
    loadSave(task, auxiliaries, Id_Node_ReferenceAngle, data._referenceAngle, defaultObject._referenceAngle);
    loadSave(task, auxiliaries, Id_Node_Color, data._color, defaultObject._color);
    loadSave(task, auxiliaries, Id_Node_NumAdditionalConnections, data._numAdditionalConnections, defaultObject._numAdditionalConnections);
    processLoadSaveMap(task, ar, auxiliaries);
    
    // Migrated to use optional nested serialization for backward compatibility
    serializeOptionalNested(task, ar, data._neuralNetwork, defaultObject._neuralNetwork);
    serializeOptionalNested(task, ar, data._cellType, defaultObject._cellType);
    serializeOptionalNested(task, ar, data._signalRestriction, defaultObject._signalRestriction);
}
```

## Conclusion

The `serializeOptionalNested` helper provides a practical balance between backward compatibility and implementation complexity. Use it for all new nested structures and gradually migrate existing ones as they need updates.

For questions or issues, refer to:
- `SerializationBackwardCompatibilityOptions.md` - Detailed analysis of all options
- `SerializerService.cpp` - Implementation reference
- `SerializerServiceTest.cpp` - Test examples
