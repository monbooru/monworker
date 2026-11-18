package infer

import (
	"image"

	"golang.org/x/image/draw"
)

func padAndResize(src image.Image, size int, fill [3]uint8) *image.RGBA {
	scaled, offX, offY := resizeAspect(src, size)

	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	for i := 0; i < len(dst.Pix); i += 4 {
		dst.Pix[i+0] = fill[0]
		dst.Pix[i+1] = fill[1]
		dst.Pix[i+2] = fill[2]
		dst.Pix[i+3] = 0xFF
	}

	draw.Draw(dst, scaled.Bounds().Add(image.Pt(offX, offY)), scaled, image.Point{}, draw.Over)
	return dst
}

func resizeAspect(src image.Image, size int) (scaled *image.RGBA, offX, offY int) {
	b := src.Bounds()
	w, h := b.Max.X-b.Min.X, b.Max.Y-b.Min.Y
	scaleW, scaleH := size, size
	if w >= h {
		scaleH = max(1, h*size/w)
	} else {
		scaleW = max(1, w*size/h)
	}
	scaled = image.NewRGBA(image.Rect(0, 0, scaleW, scaleH))

	draw.BiLinear.Scale(scaled, scaled.Bounds(), src, b, draw.Src, nil)
	return scaled, (size - scaleW) / 2, (size - scaleH) / 2
}

func rgb8(img *image.RGBA) []byte {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	out := make([]byte, 0, w*h*3)
	for y := range h {
		row := img.Pix[y*img.Stride:]
		for x := range w {
			out = append(out, row[4*x], row[4*x+1], row[4*x+2])
		}
	}
	return out
}
