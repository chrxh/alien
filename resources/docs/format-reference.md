# Format reference

This file shows which Markdown elements the documentation viewer supports. Each chapter is a *.md* file in *resources/docs*. The file *contents.md* defines the table of contents. Each heading there starts a part and each linked list item becomes a chapter, titled with the link text. The format reference itself is meant for authors and is not listed in *contents.md*, so the documentation window does not show it.

## Headings

Each chapter starts with a level 1 heading. It is drawn in a larger bold font with a line below it. The table of contents and the search use the link text from *contents.md* as the chapter title instead.

Level 2 headings divide a chapter into sections. The table of contents lists them below the open chapter, and the search shows one result per matching section. The text before the first level 2 heading forms a section named after the chapter. Level 3 and deeper headings appear in the text but not in the table of contents. Every heading can be the target of a link.

## Text

Paragraphs wrap automatically. Inline styles: **bold**, *italic* and `inline code`. Since no italic font is available, italic text is shown in the higher-contrast color of bold text and inline code instead.
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

Columns are left-aligned by default. A column written as `---:` shows its cells right-aligned, while its header stays left-aligned. Such columns are intended for pure numbers. Columns written as `:---:` are not centered but shown left-aligned.

| Setting | Unit | Value |
| --- | --- | ---: |
| Time step size | | 1.0 |
| Radius | px | 64 |
| Energy | | 1,500 |

## Images

Images are PNG or JPEG files, referenced by a path relative to *resources/docs*. Create them at twice their display size. The viewer draws them at half their pixel size times the UI scale and shrinks them to fit the available width. The alt text is shown as a caption.

An image is shown as a block of its own. If it is part of a paragraph with other text, the paragraph is split, and the text before and after the image appears above and below it. Inside lists, tables, notes and headings, only its alt text is shown. Images are loaded when they become visible and released when another chapter is opened.

![The ALIEN logo](../images/logo.png)

## Notes

> A block quote is rendered as a highlighted note. It can contain **inline styles** and [links](#links).
>
> A note can also contain several paragraphs, lists and code blocks:
>
> - First point
> - Second point

A heading, table or horizontal rule cannot be part of a note. Inside a block quote, it ends the note, and the rest of the quote continues in a new note.

## Code blocks

Code blocks are shown in a monospace font inside a box. Their lines do not wrap, so text beyond the right edge of the box is cut off.

```
EngineTests.exe --gtest_filter=Suite.Test
```

## Links

- Link to another chapter: [How life works](how-life-works.md)
- Link to a section of another chapter: [Sensor](cell-types.md#sensor)
- Link to a section in the same file: [Tables](#tables)
- External link: [alien-project.org](https://alien-project.org)
- Bare web address, linked automatically: https://alien-project.org

External links open in the default web browser.

---

Anchors are derived from headings as on GitHub: the text is converted to lower case, each space becomes a hyphen, and punctuation other than hyphens and underscores is dropped. A repeated heading gets a number, for example `#signals-1` for the second heading named "Signals".
