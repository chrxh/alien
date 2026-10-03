# Format reference

This chapter shows which Markdown elements the documentation viewer supports. Each chapter is a `.md` file in `resources/docs`. Chapters are ordered by file name, and the first top-level heading becomes the chapter title.

## Text

Paragraphs wrap automatically. Inline styles: **bold**, *italic* and `inline code`.
A line ending with a backslash\
forces a line break.

## Lists

- Bullet item
- Another bullet item
  - Nested item
  - Another nested item
- Back on the first level

1. First step
2. Second step
3. Third step

## Tables

Right-aligned columns are written as `---:` and are intended for pure numbers.

| Setting | Unit | Value |
| --- | --- | ---: |
| Time step size | | 1.0 |
| Radius | px | 64 |
| Energy | | 1,500 |

## Images

Images are PNG files, referenced relative to `resources/docs`. They are drawn at their pixel size (multiplied by the UI scale) and shrunk to the available width. The alt text is shown as a caption.

![The ALIEN logo](../images/logo.png)

## Notes

> A block quote is rendered as a highlighted note. It can contain **inline styles** and [links](#links).

## Code blocks

```
EngineTests.exe --gtest_filter=Suite.Test
```

## Links

- Link to another chapter: [Creatures and cells](02-creatures-and-cells.md)
- Link to a section of another chapter: [Cell types](02-creatures-and-cells.md#cell-types)
- Link to a section in this chapter: [Tables](#tables)
- External link: [alien-project.org](https://alien-project.org)

---

Anchors are derived from headings like on GitHub: lower case, spaces become dashes, punctuation is dropped.
