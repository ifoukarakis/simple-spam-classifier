package gr.ifoukarakis.spam;

public class Email {
    private String body;
    private int label;

    public Email(String body, int label) {
        this.body = body;
        this.label = label;
    }

    public String getBody() {
        return body;
    }

    public void setBody(String body) {
        this.body = body;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }
}
