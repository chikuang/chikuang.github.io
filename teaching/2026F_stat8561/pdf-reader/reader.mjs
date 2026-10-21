const status = document.getElementById('status');
const previous = document.getElementById('previous');
const next = document.getElementById('next');
const pageNumber = document.getElementById('page-number');
const pageCount = document.getElementById('page-count');
const zoom = document.getElementById('zoom');
function fail(error) {
  console.error('Template PDF could not be displayed:', error);
  status.classList.add('error');
  status.textContent = 'The PDF could not be displayed. Reload this page or use Download PDF.';
}
try {
  // Keep the renderer and viewer on the same pinned, locally hosted version.
  const pdfjs = await import('./vendor/pdf.mjs');
  const { EventBus, PDFLinkService, PDFViewer } = await import('./vendor/pdf_viewer.mjs');
  pdfjs.GlobalWorkerOptions.workerSrc = new URL('./vendor/pdf.worker.mjs', import.meta.url).href;
  const eventBus = new EventBus();
  const linkService = new PDFLinkService({ eventBus });
  const viewer = new PDFViewer({
    container: document.getElementById('viewerContainer'), eventBus, linkService,
    annotationMode: pdfjs.AnnotationMode.ENABLE,
    annotationEditorMode: pdfjs.AnnotationEditorType.DISABLE,
    imageResourcesPath: new URL('./vendor/images/', import.meta.url).href,
  });
  linkService.setViewer(viewer);
  const updateControls = () => {
    pageNumber.value = viewer.currentPageNumber;
    previous.disabled = viewer.currentPageNumber <= 1;
    next.disabled = viewer.currentPageNumber >= viewer.pagesCount;
  };
  eventBus.on('pagesinit', () => {
    viewer.currentScaleValue = 'page-width';
    pageNumber.max = viewer.pagesCount;
    pageCount.textContent = `of ${viewer.pagesCount}`;
    pageNumber.disabled = zoom.disabled = false;
    updateControls();
  });
  eventBus.on('pagechanging', updateControls);
  eventBus.on('pagerendered', ({ error }) => {
    if (error) { fail(error); return; }
    status.textContent = `${viewer.pagesCount}-page template outline · Finished report: 8–10 pages of main text.`;
  });
  previous.addEventListener('click', () => { viewer.currentPageNumber -= 1; });
  next.addEventListener('click', () => { viewer.currentPageNumber += 1; });
  const goToPage = () => {
    const target = Number(pageNumber.value);
    if (Number.isInteger(target) && target >= 1 && target <= viewer.pagesCount) {
      viewer.currentPageNumber = target;
    }
  };
  pageNumber.addEventListener('input', goToPage);
  pageNumber.addEventListener('change', () => { goToPage(); updateControls(); });
  pageNumber.addEventListener('keydown', event => {
    if (event.key === 'Enter') { goToPage(); updateControls(); }
  });
  zoom.addEventListener('change', () => { viewer.currentScaleValue = zoom.value; });
  const task = pdfjs.getDocument({
    url: new URL('../project-report-template.pdf', import.meta.url).href,
    standardFontDataUrl: new URL('./vendor/standard_fonts/', import.meta.url).href,
    isEvalSupported: false,
  });
  const pdf = await task.promise;
  viewer.setDocument(pdf);
  linkService.setDocument(pdf);
} catch (error) { fail(error); }
