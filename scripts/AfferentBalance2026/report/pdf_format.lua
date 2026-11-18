-- PDF-only formatting: keep the Markdown and its numerical results unchanged.
function RawBlock(el)
  if el.format == 'html' then
    local id = el.text:match('^<a id="([%w%-]+)"></a>$')
    if id then
      return pandoc.RawBlock('latex', '\\hypertarget{' .. id .. '}{}')
    end
  end
end

function Code(el)
  if not el.text:find('[{}\n]') then
    return pandoc.RawInline('latex', '\\nolinkurl{' .. el.text .. '}')
  end
end

function CodeBlock(el)
  return pandoc.RawBlock('latex',
    '\\begin{Verbatim}[fontsize=\\footnotesize,breaklines,breakanywhere]\n'
    .. el.text .. '\n\\end{Verbatim}')
end

function Math(el)
  -- Cyrillic units must use the text font, not the Latin math alphabet.
  el.text = el.text:gsub('{\\rm ([^{}]+)}', function(s)
    if s:find('[\128-\255]') then
      return '\\text{' .. s .. '}'
    end
    return '{\\rm ' .. s .. '}'
  end)
  return el
end

function Table(el)
  -- Seven rounded 1/7 widths exceed the line by 0.03%; use exact widths
  -- and leave room for experiment names in the first column.
  if #el.colspecs == 7 then
    local widths = {0.22, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13}
    local specs = el.colspecs
    for i, width in ipairs(widths) do specs[i][2] = width end
    el.colspecs = specs
  end
  -- Give genome names space without wasting a sixth of the page on indices.
  if pandoc.utils.stringify(el):find('Индекс', 1, true) then
    local widths = nil
    if #el.colspecs == 6 then
      widths = {0.07, 0.29, 0.17, 0.09, 0.19, 0.19}
    elseif #el.colspecs == 3 then
      widths = {0.30, 0.53, 0.17}
    end
    if widths then
      local specs = el.colspecs
      for i, width in ipairs(widths) do specs[i][2] = width end
      el.colspecs = specs
    end
  end
  return {
    pandoc.RawBlock('latex', '\\begingroup\\small'),
    el,
    pandoc.RawBlock('latex', '\\endgroup')
  }
end
