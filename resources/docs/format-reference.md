# Format reference

This chapter shows which Markdown elements the documentation viewer supports. Each chapter is a `.md` file in `resources/docs`. The file `contents.md` defines the table of contents. Each heading there starts a part and each linked list item becomes a chapter, titled with the link text.

## Text

Paragraphs wrap automatically. Inline styles: **bold**, *italic* and `inline code`. Italic text is shown in a brighter color, since no italic font is available.
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

   A list item can contain further paragraphs and code blocks. They are indented like the text of the item.

   ```
   Code inside a list item
   ```

3. Third step

## Tables

Columns are left aligned. Right-aligned columns are written as `---:` and are intended for pure numbers.

| Setting | Unit | Value |
| --- | --- | ---: |
| Time step size | | 1.0 |
| Radius | px | 64 |
| Energy | | 1,500 |

## Images

Images are PNG or JPEG files, referenced relative to `resources/docs`. They are prepared at twice their display size. The viewer draws them at half their pixel size, multiplied by the UI scale, and shrinks them to the available width. The alt text is shown as a caption.

An image must stand in its own paragraph. Inside lists, tables and notes, only its alt text is shown. Images are loaded when they become visible and released when another chapter is opened.

![The ALIEN logo](../images/logo.png)

## Notes

> A block quote is rendered as a highlighted note. It can contain **inline styles** and [links](#links).
>
> A note can also contain several paragraphs and lists:
>
> - First point
> - Second point

## Code blocks

```
EngineTests.exe --gtest_filter=Suite.Test
```

## Links

- Link to another chapter: [How life works](how-life-works.md)
- Link to a section of another chapter: [Sensor](cell-types.md#sensor)
- Link to a section in this chapter: [Tables](#tables)
- External link: [alien-project.org](https://alien-project.org)

External links open in the default web browser.

---

Anchors are derived from headings like on GitHub: lower case, spaces become dashes, punctuation is dropped. A repeated heading gets a number, for example `#signals-1` for the second heading named "Signals".
